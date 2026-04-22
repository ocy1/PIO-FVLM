#!/bin/bash
MODEL=llava-v1.5-7b
#MODEL=llava-v1.5-13b
MODEL2=llava-v1.6-vicuna-7b
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$MODEL.jsonl \
    --temperature 0 \
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --visual_token_num 576\
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$MODEL.jsonl
