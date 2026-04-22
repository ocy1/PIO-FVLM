#!/usr/bin/env python
"""
单样本速度对比：prune_1 vs prune_3 vs prune_4
加载 MME 第一条样本，分别切换 dispatch，测速。

用法：
    cd /home/disk/vscan_current_layer
    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.bench_prune_speed
"""
import os, sys, json, time, math
import torch
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# ============ 配置（和 mme.sh 一致）============
MODEL_PATH       = "liuhaotian/llava-v1.6-vicuna-7b"
QUESTION_FILE    = "./playground/data/eval/MME/llava_mme.jsonl"
IMAGE_FOLDER     = "./playground/data/eval/MME/MME_Benchmark_release_version"
CONV_MODE        = "vicuna_v1"
LAYER_LIST       = [1, 10, 15]
IMAGE_TOKEN_LIST = [436, 255, 173]
VISUAL_TOKEN_NUM = 576
WARMUP  = 2
REPEATS = 5
# ================================================


def load_one_sample(tokenizer, image_processor, model_config):
    with open(os.path.expanduser(QUESTION_FILE), "r") as f:
        line = json.loads(f.readline())
    qs = DEFAULT_IMAGE_TOKEN + '\n' + line["text"]
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_pil = Image.open(os.path.join(IMAGE_FOLDER, line["image"])).convert("RGB")
    image_tensor, images_pil = process_images([image_pil], image_processor, model_config)
    image_tensor = image_tensor[0]

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda')
    return input_ids, image_tensor, [image_pil.size]


def run_one(model, input_ids, image_tensor, image_sizes):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=128,
            use_cache=True,
        )
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    print("=" * 60)
    print("Prune Speed Benchmark (single sample, MME)")
    print("=" * 60)

    disable_torch_init()
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name,
        True,  # llm_pruning
        use_flash_attn=False,
        visual_token_num=VISUAL_TOKEN_NUM,
        selected_indices=[],
    )
    llama_model = model.model  # LlamaModel_X instance
    llama_model.layer_list = LAYER_LIST
    llama_model.image_token_list = [VISUAL_TOKEN_NUM] + IMAGE_TOKEN_LIST

    input_ids, image_tensor, image_sizes = load_one_sample(
        tokenizer, image_processor, model.config
    )

    # ---- 保存原始方法引用 ----
    orig_prune_1 = llama_model.layer_prune
    orig_prune_3 = llama_model.layer_prune_3
    orig_prune_4 = llama_model.layer_prune_4

    # ---- 测试配置 ----
    # dispatch 里 prune_4 列表优先于 prune_3。
    # 我们通过 monkey-patch self.layer_prune_3 / self.layer_prune_4 / self.layer_prune
    # 全部指向同一个目标函数，这样不管 dispatch 走哪个分支都会调到我们想测的。
    configs = [
        ("prune_1 (gradient+NMS)",    orig_prune_1),
        ("prune_3 (SVD+Tikhonov)",    orig_prune_3),
        ("prune_4 (ridge+GPU greedy)", orig_prune_4),
    ]

    results = {}
    for name, target_func in configs:
        # monkey-patch: 所有分支都指向 target_func
        llama_model.layer_prune   = target_func
        llama_model.layer_prune_3 = target_func
        llama_model.layer_prune_4 = target_func

        # warmup
        print(f"\n--- {name} ---")
        for w in range(WARMUP):
            t = run_one(model, input_ids, image_tensor, image_sizes)
            print(f"  warmup {w}: {t*1000:.1f} ms")

        # timed
        times = []
        for r in range(REPEATS):
            t = run_one(model, input_ids, image_tensor, image_sizes)
            times.append(t)
            print(f"  run {r}: {t*1000:.1f} ms")

        times.sort()
        median = times[len(times) // 2]
        results[name] = median
        print(f"  >> median = {median*1000:.1f} ms")

    # 恢复
    llama_model.layer_prune   = orig_prune_1
    llama_model.layer_prune_3 = orig_prune_3
    llama_model.layer_prune_4 = orig_prune_4

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print(f"{'Method':40s} {'Median(ms)':>10s} {'vs prune_1':>12s}")
    print("-" * 62)
    baseline = None
    for name, ms_sec in results.items():
        ms = ms_sec * 1000
        if baseline is None:
            baseline = ms
            print(f"  {name:38s} {ms:10.1f}   {'baseline':>12s}")
        else:
            ratio = baseline / ms if ms > 0 else float('inf')
            print(f"  {name:38s} {ms:10.1f}   {ratio:10.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
