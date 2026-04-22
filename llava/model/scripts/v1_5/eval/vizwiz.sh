#!/bin/bash
MODEL=llava-v1.5-7b
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/$MODEL.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/$MODEL.json
