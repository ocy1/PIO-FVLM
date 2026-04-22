import os
import argparse
import json
import math

# =========================================================
# 0) 先解析 --gpu，并在 import torch/transformers 前设置可见GPU
# =========================================================
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=int, default=3, help="指定物理GPU编号，例如 4")
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

# =========================================================
# 1) 再导入 torch / transformers / 其他重库
# =========================================================
import torch
from tqdm import tqdm
import shortuuid

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

    # Change to custom forward function
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune

    # 与“刚刚那份逻辑”一致：默认不 patch FlashAttention2，避免退化变慢
    if args.patch_flash_attn:
        Qwen2_5_VLVisionFlashAttention2.forward = Qwen2_5_VLVisionFlashAttention2_X.forward
        print("[INFO] FlashAttention2 forward is patched (args.patch_flash_attn=True).")
    else:
        print("[INFO] FlashAttention2 forward is NOT patched (default).")

    # 强制单卡（可见卡的 cuda:0）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": 0},
    )
    model.eval()

    model.model.layer_list = eval(args.layer_list)
    model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
    model.image_token_ratio = args.image_token_ratio

    # processor（与你基线一致）
    min_pixels = 1008*1008
    max_pixels = 1008*1008  
    # min_pixels = 600*600
    # max_pixels = 600*600
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    # ---- Data ----
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    # generation kwargs（与你基线一致：避免传 None）
    gen_kwargs = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams}
    if args.temperature is not None:
        gen_kwargs["temperature"] = float(args.temperature)
        gen_kwargs["do_sample"] = True if args.temperature > 0 else False
    if args.top_p is not None:
        gen_kwargs["top_p"] = float(args.top_p)

    # 可选：只打印一次 device 信息（避免每题 print 导致巨慢）
    printed_debug = False

    for line in tqdm(questions):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        # if args.single_pred_prompt:
        #     suffix = "Answer with the option's letter from the given choices directly."
        #     qs = qs + "\n" + suffix
        #     cur_prompt = cur_prompt + "\n" + suffix
        suffix = "Answer with the option's letter from the given choices directly."
        qs = qs + "\n" + suffix
        cur_prompt = cur_prompt + "\n" + suffix

        if "image" in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": qs},
                    ],
                }
            ]
            cur_prompt = "<image>\n" + cur_prompt
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": qs}],
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

        if args.debug_device and (not printed_debug):
            print(
                f"[DEBUG] model on {next(model.parameters()).device}, "
                f"input_ids on {inputs.input_ids.device}"
            )
            printed_debug = True

        # with torch.inference_mode():
        #     generated_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = output_text[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 与基线一致：支持显式指定物理 GPU（必须放在脚本最前面的预解析已生效）
    parser.add_argument("--gpu", type=int, default=5, help="指定物理GPU编号，例如 4")

    parser.add_argument("--model-path", type=str, default="qwen_model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/scienceqa/images/test")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/scienceqa/llava_test_CQM-A.json")
    parser.add_argument("--answers-file", type=str, default="./playground/data/eval/scienceqa/answers/Qwen2.5-VL-7B-Instruct.jsonl")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=16)

    parser.add_argument("--layer-list", type=str, default="[1,8,14]")
    parser.add_argument("--image-token-ratio-list", type=str, default="[0.6003, 0.3519, 0.1436]")
    # parser.add_argument("--layer-list", type=str, default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]")
    # parser.add_argument("--image-token-ratio-list", type=str, default="[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]")
    parser.add_argument("--image-token-ratio", type=float, default=1.0)

    # 与基线一致：默认不 patch FlashAttention2
    parser.add_argument("--patch-flash-attn", action="store_true")

    # 可选：只打印一次设备信息
    parser.add_argument("--debug-device", action="store_true")

    args = parser.parse_args()
    eval_model(args)
