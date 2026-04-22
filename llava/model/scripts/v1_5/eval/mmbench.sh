#!/bin/bash
MODEL=llava-v1.5-7b
SPLIT="mmbench_dev_20230712"
MODEL2=llava-v1.6-vicuna-7b
python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$MODEL.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --visual_token_num 576\
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL
