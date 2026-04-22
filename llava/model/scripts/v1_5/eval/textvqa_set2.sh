#!/bin/bash
MODEL=llava-v1.5-7b
MODEL2=llava-v1.6-vicuna-7b
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/textvqa/subset/set2_questions.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/set2_answers.jsonl \
    --temperature 0 \
    --visual_token_num 576\
    --layer_list       '[1,10,15]'\
    --image_token_list '[92, 54, 22]'\
    --conv-mode vicuna_v1
