#!/bin/bash
MODEL=llava-v1.5-7b
#MODEL=llava-v1.5-13b
MODEL2=llava-v1.6-vicuna-7b
python -m llava.eval.model_vqa_science \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$MODEL.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --visual_token_num 576\
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$MODEL.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl  \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result.json
