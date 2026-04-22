#!/bin/bash
MODEL=llava-v1.5-7b
#MODEL=llava-v1.5-13b
MODEL2=llava-v1.6-vicuna-7b
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

#q  PT="llava-v1.5-7b"
CKPT="llava-v1.5-7b"
#CKPT="llava-v1.5-13b"
SPLIT="llava_vqav2_mscoco_test-dev2015"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path liuhaotian/$MODEL \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder ./playground/data/eval/vqav2/test2015 \
        --answers-file ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --visual_token_num 576 \
        --layer_list       '[1,10,15]'\
        --image_token_list '[218, 128, 54]'\
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

