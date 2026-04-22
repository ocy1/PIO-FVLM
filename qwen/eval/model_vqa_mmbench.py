import os
import argparse
import json
import math
import base64
from io import BytesIO

# =========================================================
# 0) 先解析 --gpu，并在 import torch/transformers 前设置可见GPU
# =========================================================
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=int, default=7, help="指定物理GPU编号，例如 4")
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

# =========================================================
# 1) 再导入 torch / transformers / 其他重库
# =========================================================
import torch
import pandas as pd
from tqdm import tqdm
import shortuuid

from PIL import Image

from transformers import AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VLVisionSdpaAttention,
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionFlashAttention2,
)

from qwen.model.qwen2_5_vl_custom import (
    Qwen2_5_VLForConditionalGeneration_X,
    Qwen2_5_VisionTransformerPretrainedModel_X,
    Qwen2_5_VLVisionBlock_X,
    Qwen2_5_VLVisionSdpaAttention_X,
    Qwen2_5_VisionPatchEmbed_X,
    Qwen2_5_VLModel_X,
    Qwen2_5_VLVisionFlashAttention2_X,
)

from qwen_vl_utils import process_vision_info

all_options = ["A", "B", "C", "D"]


def load_image_from_base64(image_b64: str) -> Image.Image:
    # 注意：这里不做额外 copy，尽量省开销
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    return model_paths[-1]


def is_none(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ("nan", "none"):
        return True
    return False


def get_options_from_row(row, option_keys):
    parsed = []
    for k in option_keys:
        v = getattr(row, k)
        if is_none(v):
            break
        parsed.append(v)
    return parsed


def eval_model(args):
    # ---- GPU info ----
    if torch.cuda.is_available():
        print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[INFO] torch sees {torch.cuda.device_count()} GPU(s)")
        print(f"[INFO] cuda:0 => {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # patch 自定义 forward
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune

    # ⚠️ 默认不 patch FlashAttention2（很多人这里 patch 后会退化导致巨慢）
    if args.patch_flash_attn:
        Qwen2_5_VLVisionFlashAttention2.forward = Qwen2_5_VLVisionFlashAttention2_X.forward
        print("[INFO] FlashAttention2 forward is patched (args.patch_flash_attn=True).")
    else:
        print("[INFO] FlashAttention2 forward is NOT patched (default).")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": 0},  # 强制单卡：可见卡的 cuda:0
    )
    model.eval()

    model.model.layer_list = eval(args.layer_list)
    model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
    model.image_token_ratio = args.image_token_ratio
    
    min_pixels = 1008*1008
    max_pixels = 1008*1008
    # min_pixels = 600*600
    # max_pixels = 600*600
    



    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # ---- Data ----
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    # generation kwargs（避免传 None）
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}
    if args.temperature is not None:
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["do_sample"] = True if args.temperature > 0 else False
    if args.top_p is not None:
        gen_kwargs["top_p"] = float(args.top_p)

    # =========================================================
    # 2) 关键提速点：用 itertuples() 代替 iterrows()
    # =========================================================
    # itertuples 会返回 namedtuple，字段名来自列名
    for row in tqdm(questions.itertuples(index=False), total=len(questions)):
        # options / rounds
        options = get_options_from_row(row, all_options)
        cur_option_char = all_options[: len(options)]

        num_rounds = len(options) if args.all_rounds else 1

        # =========================================================
        # 3) 关键提速点：base64->PIL 解码只做一次（每题一次）
        # =========================================================
        # 这一张图会被多轮复用（all_rounds），因此放到 round 循环外
        img = load_image_from_base64(getattr(row, "image"))

        idx = getattr(row, "index")
        question = getattr(row, "question")
        hint = getattr(row, "hint")

        # 先拼一次题干（每轮会旋转 options，所以 round 内再拼选项）
        base_q = question
        if not is_none(hint):
            base_q = str(hint) + "\n" + str(question)

        for round_idx in range(num_rounds):
            # 拼当前轮的 prompt（包含选项）
            qs = base_q
            for option_char, option in zip(all_options[: len(options)], options):
                qs += "\n" + option_char + ". " + str(option)

            # if args.single_pred_prompt:
            #     if args.lang == "cn":
            #         qs += "\n请直接回答选项字母。"
            #     else:
            #         qs += "\nAnswer with the option's letter from the given choices directly."
            if args.lang == "cn":
                qs += "\n请直接回答选项字母。"
            else:
                qs += "\nAnswer with the option's letter from the given choices directly."

            cur_prompt = qs  # 写文件用

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": qs},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # with torch.inference_mode():
            #     generated_ids = model.generate(**inputs, **gen_kwargs)
            generated_ids = model.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outputs = output_text[0].strip()

            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "round_id": round_idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "options": options,
                        "option_char": cur_option_char,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()

            # rotate options（保持你原来的评测逻辑）
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 与代码1一致：支持显式指定物理 GPU
    parser.add_argument("--gpu", type=int, default=7, help="指定物理GPU编号，例如 4")

    parser.add_argument("--model-path", type=str, default="qwen_model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/mmbench/mmbench_dev_20230712.tsv")
    parser.add_argument("--answers-file",type=str,default="./playground/data/eval/mmbench/answers/mmbench_dev_20230712/Qwen2.5-VL-7B-Instruct.jsonl",)
    
    # parser.add_argument("--image-folder", type=str, default="")
    # parser.add_argument("--question-file", type=str, default="./playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv")
    # parser.add_argument("--answers-file",type=str,default="./playground/data/eval/mmbench_cn/answers/mmbench_dev_cn_20231003/Qwen2.5-VL-7B-Instruct.jsonl",)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    parser.add_argument("--layer-list", type=str, default="[1,8,14]")
    parser.add_argument("--image-token-ratio-list", type=str, default="[0.6003, 0.3519, 0.1436]")
    # parser.add_argument("--layer-list", type=str, default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]")
    # parser.add_argument("--image-token-ratio-list", type=str, default="[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]")
    parser.add_argument("--image-token-ratio", type=float, default=1.0)

    # 新增：是否 patch Vision FlashAttention2（默认关闭以避免退化变慢）
    parser.add_argument("--patch-flash-attn", action="store_true")

    args = parser.parse_args()
    eval_model(args)
