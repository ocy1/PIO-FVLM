#!/bin/bash
# ============================================================
# 实验 2: Wrong-Image Control — MME
# ============================================================
set -e
cd /home/disk/vscan_current_layer

MODEL=llava-v1.5-7b
MODEL_PATH=liuhaotian/$MODEL
CONV_MODE=vicuna_v1

DROP_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"

QUESTION_FILE=./playground/data/eval/MME/llava_mme.jsonl
IMAGE_FOLDER=./playground/data/eval/MME/MME_Benchmark_release_version
RESULT_DIR=./playground/data/eval/MME/answers_wrong_image

mkdir -p $RESULT_DIR

# ============================================================
# 一次性跑完: JS divergence + 生成答案
# ============================================================
echo "========== Running Wrong-Image Control (JS div + generate) =========="
python eval_wrong_image.py \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --question-file $QUESTION_FILE \
    --output-dir $RESULT_DIR \
    --conv-mode $CONV_MODE \
    --num-samples 0 \
    --drop-layers "$DROP_LAYERS" \
    --generate-answers \
    --max_new_tokens 128 \
    --seed 42

echo "========== Forward done. Now running MME eval per layer =========="

# ============================================================
# 提取每层答案 → MME 评估 (wrong_drop + correct_drop)
# ============================================================
DETAIL_FILE=${RESULT_DIR}/results_detail.jsonl

for DL in ${DROP_LAYERS//,/ }; do
    echo ">>> Evaluating drop_layer = $DL"

    for COND in wrong correct; do
        ANSWER_KEY="answer_${COND}_drop_${DL}"
        ANSWERS_FILE=./playground/data/eval/MME/answers/${MODEL}_${COND}_drop${DL}.jsonl
        mkdir -p ./playground/data/eval/MME/answers

        python -c "
import json
with open('${DETAIL_FILE}') as f:
    for line in f:
        d = json.loads(line)
        ans = d.get('${ANSWER_KEY}', '')
        if not ans:
            continue
        out = {
            'question_id': d['question_id'],
            'prompt': d['question'],
            'text': ans,
            'model_id': '${MODEL}_${COND}_drop${DL}',
            'metadata': {}
        }
        print(json.dumps(out))
" > $ANSWERS_FILE

        cd ./playground/data/eval/MME
        python convert_answer_to_mme.py --experiment ${MODEL}_${COND}_drop${DL}
        cd eval_tool
        echo "--- MME score (${COND} image, drop_layer=$DL) ---"
        python calculation.py --results_dir answers/${MODEL}_${COND}_drop${DL} \
            | tee /home/disk/vscan_current_layer/${RESULT_DIR}/mme_score_${COND}_drop${DL}.txt
        cd /home/disk/vscan_current_layer
    done
done

echo "========== MME Wrong-Image Control done! =========="
