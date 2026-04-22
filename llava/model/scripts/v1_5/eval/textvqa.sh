#!/bin/bash
MODEL=llava-v1.5-7b
#MODEL=llava-v1.5-13b
MODEL2=llava-v1.6-vicuna-7b
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl \
    --temperature 0 \
    --visual_token_num 576\
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl
