import os
import argparse
import math
import json

# =========================================================
# 0) 先解析 --gpu，并在 import torch/transformers 前设置可见GPU
# =========================================================
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", type=int, default=6, help="指定物理GPU编号，例如 4")
_pre_args, _ = _pre_parser.parse_known_args()

if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)

# =========================================================
# 1) 再导入 torch / transformers
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
)

from qwen.model.qwen2_5_vl_custom import (
    Qwen2_5_VLForConditionalGeneration_X,
    Qwen2_5_VisionTransformerPretrainedModel_X,
    Qwen2_5_VLVisionBlock_X,
    Qwen2_5_VLVisionSdpaAttention_X,
    Qwen2_5_VisionPatchEmbed_X,
    Qwen2_5_VLModel_X,
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
    else:
        return model_paths[-1]


def eval_model(args):
    # 只要你传了 --gpu，这里 torch 看到的 cuda:0 就是 “那张物理卡”
    if torch.cuda.is_available():
        print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[INFO] torch sees {torch.cuda.device_count()} GPU(s)")
        print(f"[INFO] cuda:0 => {torch.cuda.get_device_name(0)}")
        torch.cuda.set_device(0)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    # Change to custom forward function
    Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_X.forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = Qwen2_5_VisionTransformerPretrainedModel_X.forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_X.forward
    Qwen2_5_VLVisionSdpaAttention.forward = Qwen2_5_VLVisionSdpaAttention_X.forward
    Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbed_X.forward
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_X.forward
    Qwen2_5_VLModel.layer_prune = Qwen2_5_VLModel_X.layer_prune

    # =========================================================
    # 关键：强制整个模型放在 cuda:0（即你指定的那张物理GPU）
    # =========================================================
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map={"": 0},  # 强制单卡
    )

    model.model.layer_list = eval(args.layer_list)
    model.model.image_token_ratio_list = eval(args.image_token_ratio_list)
    model.image_token_ratio = args.image_token_ratio
    min_pixels = 1008*1008
    max_pixels = 1008*1008
    # min_pixels = 256*28*28
    # max_pixels = 1024*28*28  
    # min_pixels = 700*700
    # max_pixels = 700*700
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for line in tqdm(questions):
        cur_prompt = line["text"]
        image_path = os.path.join(args.image_folder, line["image"])
        idx = line["question_id"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": cur_prompt},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)  # 也强制把 inputs 放到 cuda:0
# =======================================================
        # [修改] 精确打印 Image Token 数量
        # =======================================================
        # 1. 获取模型配置中定义的 image token id
        img_id = model.config.image_token_id
        
        # 2. 统计 input_ids 中该 id 出现的次数
        # inputs.input_ids 是一个 tensor，我们生成布尔掩码然后求和
        num_image_tokens = (inputs.input_ids == img_id).sum().item()
        
        # 3. (可选) 同时也打印一下总 Token 数做对比
        total_tokens = inputs.input_ids.shape[1]

        print(f"--------")
        print(f"[INFO] Question ID: {idx}")
        print(f"[INFO] Image Tokens Only: {num_image_tokens}")  # <--- 您需要的数值
        print(f"[INFO] Total Tokens (Text+Img): {total_tokens}")
        
        # 验证一下：这应该等于 grid_thw 的 h/2 * w/2
        if "image_grid_thw" in inputs:
             print(f"[INFO] Image Grid Info: {inputs['image_grid_thw']}")
        print(f"--------")
        # =======================================================

        # with torch.inference_mode():
        #     generated_ids = model.generate(
        #         **inputs,
        #         max_new_tokens=args.max_new_tokens,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         do_sample=True if args.temperature > 0 else False,
        #     )
        #     generated_ids_trimmed = [
        #         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        #     ]
      
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            do_sample=True if args.temperature > 0 else False,
        )
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

    # 新增：强制单卡（物理GPU编号）
    parser.add_argument("--gpu", type=int, default=None, help="指定物理GPU编号，例如 4")
    parser.add_argument("--model-path", type=str, default="qwen_model/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-folder", type=str, default="./playground/data/eval/MME/MME_Benchmark_release_version")
    # parser.add_argument("--question-file", type=str, default="./playground/data/eval/MME/llava_mme.jsonl")
    # parser.add_argument("--answers-file", type=str, default="./playground/data/eval/MME/answers/Qwen2.5-VL-7B-Instruct.jsonl")
    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/textvqa/train_images")
    parser.add_argument("--question-file", type=str, default="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl")
    parser.add_argument("--answers-file", type=str, default="./playground/data/eval/textvqa/answers/Qwen2.5-VL-7B-Instruct2.jsonl")
    # parser.add_argument("--image-folder", type=str, default="./playground/data/eval/pope/val2014")
    # parser.add_argument("--question-file", type=str, default="./playground/data/eval/pope/llava_pope_test.jsonl")
    # parser.add_argument("--answers-file", type=str, default="./playground/data/eval/pope/answers/Qwen2.5-VL-7B-Instruct.jsonl")
    # parser.add_argument("--image-folder", type=str, default="./playground/data/eval/vqav2/test2015")
    # parser.add_argument("--question-file", type=str, default="./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl")
    # parser.add_argument("--answers-file", type=str, default="./playground/data/eval/vqav2/answers/llava_vqav2_mscoco_test-dev2015/Qwen2.5-VL-7B-Instruct.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--layer-list", type=str, default="[1]")
    parser.add_argument("--image-token-ratio-list", type=str, default="[0.1]")
    # parser.add_argument("--layer-list", type=str, default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]")
    # parser.add_argument("--image-token-ratio-list", type=str, default="[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,]")
    parser.add_argument("--image-token-ratio", type=float, default=1.0)

    args = parser.parse_args()
    eval_model(args)
