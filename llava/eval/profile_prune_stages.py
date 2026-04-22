#!/usr/bin/env python
"""
逐阶段 profile prune_3 / prune_4 的耗时分布。
在 layer_prune_3 / layer_prune_4 的关键位置插桩，
只跑 1 个样本，打印各阶段 GPU 时间。

用法：
    cd /home/disk/vscan_current_layer
    CUDA_VISIBLE_DEVICES=0 python -m llava.eval.profile_prune_stages
"""
import os, sys, json, time, math, functools
import torch
import numpy as np
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

MODEL_PATH       = "liuhaotian/llava-v1.6-vicuna-7b"
QUESTION_FILE    = "./playground/data/eval/MME/llava_mme.jsonl"
IMAGE_FOLDER     = "./playground/data/eval/MME/MME_Benchmark_release_version"
CONV_MODE        = "vicuna_v1"
LAYER_LIST       = [1, 10, 15]
IMAGE_TOKEN_LIST = [436, 255, 173]
VISUAL_TOKEN_NUM = 576


class GpuTimer:
    """用 CUDA events 精确计时"""
    def __init__(self):
        self.records = {}
    def start(self, name):
        if name not in self.records:
            self.records[name] = {"starts": [], "ends": [], "times": []}
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        self.records[name]["starts"].append(s)
        self.records[name]["ends"].append(e)
    def stop(self, name):
        self.records[name]["ends"][-1].record()
    def sync_and_report(self):
        torch.cuda.synchronize()
        print("\n" + "=" * 70)
        print(f"{'Stage':45s} {'Count':>5s} {'Total(ms)':>10s} {'Avg(ms)':>10s}")
        print("-" * 70)
        for name, rec in self.records.items():
            times = []
            for s, e in zip(rec["starts"], rec["ends"]):
                times.append(s.elapsed_time(e))
            total = sum(times)
            avg = total / len(times) if times else 0
            print(f"  {name:43s} {len(times):5d} {total:10.2f} {avg:10.2f}")
        print("=" * 70)


# 全局 timer
timer = GpuTimer()


def make_profiled_prune_3(orig_func):
    """给 layer_prune_3 加插桩"""
    import torch.nn.functional as F

    @functools.wraps(orig_func)
    def wrapper(self, cur_num, rank_layer, features, position_ids,
                attention_mask, labels, selected_indices,
                sum_visual_attention, indices_attention):
        # 我们不重写整个函数，而是在关键点计时
        # 但由于是方法内部的局部变量，我们只能用整体计时
        timer.start(f"prune3_total_layer{rank_layer}")
        result = orig_func(
            cur_num, rank_layer, features, position_ids,
            attention_mask, labels, selected_indices,
            sum_visual_attention, indices_attention,
        )
        timer.stop(f"prune3_total_layer{rank_layer}")
        return result
    return wrapper


def make_profiled_prune_4(orig_func):
    """给 layer_prune_4 加插桩"""
    @functools.wraps(orig_func)
    def wrapper(self, cur_num, rank_layer, features, position_ids,
                attention_mask, labels, selected_indices,
                sum_visual_attention, indices_attention):
        timer.start(f"prune4_total_layer{rank_layer}")
        result = orig_func(
            cur_num, rank_layer, features, position_ids,
            attention_mask, labels, selected_indices,
            sum_visual_attention, indices_attention,
        )
        timer.stop(f"prune4_total_layer{rank_layer}")
        return result
    return wrapper


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
    with torch.no_grad():
        _ = model.generate(
            input_ids, images=image_tensor, image_sizes=image_sizes,
            do_sample=False, temperature=0, max_new_tokens=128, use_cache=True,
        )
    torch.cuda.synchronize()


def main():
    disable_torch_init()
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, True,
        use_flash_attn=False, visual_token_num=VISUAL_TOKEN_NUM, selected_indices=[],
    )
    llama_model = model.model
    llama_model.layer_list = LAYER_LIST
    llama_model.image_token_list = [VISUAL_TOKEN_NUM] + IMAGE_TOKEN_LIST

    input_ids, image_tensor, image_sizes = load_one_sample(tokenizer, image_processor, model.config)

    orig_prune_3 = llama_model.layer_prune_3
    orig_prune_4 = llama_model.layer_prune_4

    # ---- Profile prune_3 ----
    print("\n>>> Profiling prune_3 <<<")
    timer.records.clear()
    import types
    llama_model.layer_prune_3 = orig_prune_3
    llama_model.layer_prune_4 = orig_prune_3  # 都走 prune_3

    # warmup
    run_one(model, input_ids, image_tensor, image_sizes)

    # 由于我们无法在方法内部插桩（局部变量），
    # 改用 torch.cuda.Event 在 forward_x 的 prune_func 调用前后计时。
    # 最简单的方式：直接 patch forward_x
    _orig_forward_x = llama_model.forward_x.__func__  # unbound

    def patched_forward_x(self_model, *args, **kwargs):
        # 这里我们不 patch forward_x，太复杂。
        # 改用更简单的方式：直接 wrap prune 函数本身
        return _orig_forward_x(self_model, *args, **kwargs)

    # 更简单：直接 wrap layer_prune_3
    call_count = [0]
    def timed_prune_3(*a, **kw):
        call_count[0] += 1
        tag = f"prune3_call{call_count[0]}"
        timer.start(tag)
        r = orig_prune_3(*a, **kw)
        timer.stop(tag)
        return r

    llama_model.layer_prune_3 = timed_prune_3
    llama_model.layer_prune_4 = timed_prune_3
    llama_model.layer_prune = timed_prune_3

    call_count[0] = 0
    run_one(model, input_ids, image_tensor, image_sizes)
    timer.sync_and_report()

    # ---- Profile prune_4 ----
    print("\n>>> Profiling prune_4 <<<")
    timer.records.clear()

    call_count_4 = [0]
    def timed_prune_4(*a, **kw):
        call_count_4[0] += 1
        tag = f"prune4_call{call_count_4[0]}"
        timer.start(tag)
        r = orig_prune_4(*a, **kw)
        timer.stop(tag)
        return r

    llama_model.layer_prune_3 = timed_prune_4
    llama_model.layer_prune_4 = timed_prune_4
    llama_model.layer_prune = timed_prune_4

    call_count_4[0] = 0
    run_one(model, input_ids, image_tensor, image_sizes)
    timer.sync_and_report()

    # 恢复
    llama_model.layer_prune_3 = orig_prune_3
    llama_model.layer_prune_4 = orig_prune_4


if __name__ == "__main__":
    main()
