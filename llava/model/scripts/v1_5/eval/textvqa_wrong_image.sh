#!/bin/bash
# ============================================================
# 实验 2: Wrong-Image Control — TextVQA
# ============================================================
set -e
cd /home/disk/vscan_current_layer

MODEL=llava-v1.5-7b
MODEL_PATH=liuhaotian/$MODEL
CONV_MODE=vicuna_v1

DROP_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"

QUESTION_FILE=./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl
IMAGE_FOLDER=./playground/data/eval/textvqa/train_images
ANNOTATION_FILE=./playground/data/eval/textvqa/TextVQA_0.5.1_val.json
RESULT_DIR=./playground/data/eval/textvqa/answers_wrong_image

mkdir -p $RESULT_DIR

# ============================================================
# 一次性跑完: JS divergence + 生成答案 (正确图 + 错误图)
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

echo "========== Forward done. Now running TextVQA eval per layer =========="

# ============================================================
# 提取每层答案 → TextVQA 评估 (wrong_drop + correct_drop)
# ============================================================
DETAIL_FILE=${RESULT_DIR}/results_detail.jsonl

for DL in ${DROP_LAYERS//,/ }; do
    echo ">>> Evaluating drop_layer = $DL"

    # --- 错误图 drop 的答案 ---
    WRONG_FILE=${RESULT_DIR}/${MODEL}_wrong_drop${DL}.jsonl
    python -c "
import json
with open('${DETAIL_FILE}') as f:
    for line in f:
        d = json.loads(line)
        ans = d.get('answer_wrong_drop_${DL}', '')
        if not ans:
            continue
        out = {
            'question_id': d['question_id'],
            'prompt': d['question'],
            'text': ans,
            'model_id': '${MODEL}_wrong_drop${DL}',
            'metadata': {}
        }
        print(json.dumps(out))
" > $WRONG_FILE

    echo "--- TextVQA score (wrong image, drop_layer=$DL) ---"
    python -m llava.eval.eval_textvqa \
        --annotation-file $ANNOTATION_FILE \
        --result-file $WRONG_FILE \
        | tee ${RESULT_DIR}/textvqa_score_wrong_drop${DL}.txt

    # --- 正确图 drop 的答案 ---
    CORRECT_FILE=${RESULT_DIR}/${MODEL}_correct_drop${DL}.jsonl
    python -c "
import json
with open('${DETAIL_FILE}') as f:
    for line in f:
        d = json.loads(line)
        ans = d.get('answer_correct_drop_${DL}', '')
        if not ans:
            continue
        out = {
            'question_id': d['question_id'],
            'prompt': d['question'],
            'text': ans,
            'model_id': '${MODEL}_correct_drop${DL}',
            'metadata': {}
        }
        print(json.dumps(out))
" > $CORRECT_FILE

    echo "--- TextVQA score (correct image, drop_layer=$DL) ---"
    python -m llava.eval.eval_textvqa \
        --annotation-file $ANNOTATION_FILE \
        --result-file $CORRECT_FILE \
        | tee ${RESULT_DIR}/textvqa_score_correct_drop${DL}.txt

done

echo "========== TextVQA Wrong-Image Control done! =========="
