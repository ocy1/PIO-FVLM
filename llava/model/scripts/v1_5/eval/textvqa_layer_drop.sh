#!/bin/bash
# ============================================================
# 实验 1: Layer-wise Token Drop — TextVQA
# 一次运行得到两个指标:
#   1. JS divergence vs drop_layer
#   2. TextVQA accuracy vs drop_layer
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
RESULT_DIR=./playground/data/eval/textvqa/answers_layer_drop

mkdir -p $RESULT_DIR

# ============================================================
# 一次性跑完: JS divergence + 生成答案
# ============================================================
echo "========== Running layer-wise drop (JS div + generate) =========="
python eval_layer_drop.py \
    --model-path $MODEL_PATH \
    --image-folder $IMAGE_FOLDER \
    --question-file $QUESTION_FILE \
    --output-dir $RESULT_DIR \
    --conv-mode $CONV_MODE \
    --num-samples 0 \
    --drop-layers "$DROP_LAYERS" \
    --generate-answers \
    --max_new_tokens 128

echo "========== Forward done. Now running TextVQA eval per layer =========="

# ============================================================
# 提取每层答案 → TextVQA 评估
# ============================================================
DETAIL_FILE=${RESULT_DIR}/results_detail.jsonl

for DL in ${DROP_LAYERS//,/ }; do
    echo ">>> Evaluating drop_layer = $DL"
    ANSWERS_FILE=${RESULT_DIR}/${MODEL}_drop${DL}.jsonl

    python -c "
import json
with open('${DETAIL_FILE}') as f:
    for line in f:
        d = json.loads(line)
        ans = d.get('answer_drop_${DL}', '')
        if not ans:
            continue
        out = {
            'question_id': d['question_id'],
            'prompt': d['question'],
            'text': ans,
            'model_id': '${MODEL}_drop${DL}',
            'metadata': {}
        }
        print(json.dumps(out))
" > $ANSWERS_FILE

    echo "--- TextVQA score for drop_layer=$DL ---"
    python -m llava.eval.eval_textvqa \
        --annotation-file $ANNOTATION_FILE \
        --result-file $ANSWERS_FILE \
        | tee ${RESULT_DIR}/textvqa_score_drop${DL}.txt

done

# ============================================================
# 汇总 (同时打印到终端 + 写入汇总文件)
# ============================================================
SUMMARY_FILE=${RESULT_DIR}/summary_table.txt

{
echo "========== Results Summary (TextVQA Layer Drop) =========="
echo ""

echo "==================== 汇总表格 ===================="
printf "%-12s | %-14s | %-10s | %-14s\n" "drop_layer" "JS Div(mean)" "JS Div(std)" "TextVQA Acc"
printf "%s\n" "-------------|----------------|------------|---------------"

python -c "
import json, re, os

# 读取 JS divergence
js_path = '${RESULT_DIR}/js_divergence_summary.json'
js_data = {}
if os.path.exists(js_path):
    with open(js_path) as f:
        js_data = json.load(f)

layers = '${DROP_LAYERS}'.split(',')
for dl in layers:
    js_mean = js_data.get(dl, {}).get('mean', float('nan'))
    js_std  = js_data.get(dl, {}).get('std', float('nan'))

    acc = '-'
    score_file = '${RESULT_DIR}/textvqa_score_drop' + dl + '.txt'
    if os.path.exists(score_file):
        with open(score_file) as sf:
            for line in sf:
                m = re.search(r'[Aa]ccuracy\D*([\d.]+)', line)
                if m:
                    acc = m.group(1)

    print(f'{dl:<12} | {js_mean:<14.6f} | {js_std:<10.6f} | {acc:<14}')
"

echo ""
echo "========== All done! =========="
} 2>&1 | tee $SUMMARY_FILE
