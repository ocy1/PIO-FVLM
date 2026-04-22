#!/bin/bash
MODEL=llava-v1.5-7b
python -m llava.eval.model_vqa \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --temperature 0 \
    --visual_token_num 576 \
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$MODEL.json

