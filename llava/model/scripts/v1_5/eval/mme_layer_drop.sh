#!/bin/bash
# ============================================================
# 实验 1: Layer-wise Token Drop — MME
# 一次运行得到两个指标:
#   1. JS divergence vs drop_layer  (forward-only, 快)
#   2. MME accuracy vs drop_layer   (需要 generate, 慢)
# ============================================================
set -e
cd /home/disk/vscan_current_layer

MODEL=llava-v1.5-7b
MODEL_PATH=liuhaotian/$MODEL
CONV_MODE=vicuna_v1

# drop layer 列表 (0=no-image, 32=full, 即不删除)
DROP_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32"

# 数据路径
QUESTION_FILE=./playground/data/eval/MME/llava_mme.jsonl
IMAGE_FOLDER=./playground/data/eval/MME/MME_Benchmark_release_version
RESULT_DIR=./playground/data/eval/MME/answers_layer_drop

mkdir -p $RESULT_DIR

# ============================================================
# 一次性跑完: JS divergence + 生成答案 (模型只加载一次)
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

echo "========== Forward done. Now running MME eval per layer =========="

# ============================================================
# 从 results_detail.jsonl 提取每层答案 → MME 评估
# ============================================================
DETAIL_FILE=${RESULT_DIR}/results_detail.jsonl

for DL in ${DROP_LAYERS//,/ }; do
    echo ">>> Evaluating drop_layer = $DL"

    # 提取该层的答案为 model_vqa_loader 兼容格式
    ANSWER_KEY="answer_drop_${DL}"
    # 写到 MME 标准 answers 目录
    ANSWERS_FILE=./playground/data/eval/MME/answers/${MODEL}_drop${DL}.jsonl
    mkdir -p ./playground/data/eval/MME/answers

    python -c "
import json, sys
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
            'model_id': '${MODEL}_drop${DL}',
            'metadata': {}
        }
        print(json.dumps(out))
" > $ANSWERS_FILE

    # convert + calculate
    cd ./playground/data/eval/MME
    python convert_answer_to_mme.py --experiment ${MODEL}_drop${DL}
    cd eval_tool
    echo "--- MME score for drop_layer=$DL ---"
    python calculation.py --results_dir answers/${MODEL}_drop${DL} | tee /home/disk/vscan_current_layer/${RESULT_DIR}/mme_score_drop${DL}.txt
    cd /home/disk/vscan_current_layer
done

# ============================================================
# 汇总 (同时打印到终端 + 写入汇总文件)
# ============================================================
SUMMARY_FILE=${RESULT_DIR}/summary_table.txt

{
echo "========== Results Summary (MME Layer Drop) =========="
echo "JS Divergence: ${RESULT_DIR}/js_divergence_summary.json"
echo "Per-sample detail: ${RESULT_DIR}/results_detail.jsonl"
echo "MME scores: ${RESULT_DIR}/mme_score_drop*.txt"
echo ""

echo "==================== 汇总表格 ===================="
printf "%-12s | %-14s | %-10s | %-14s | %-14s\n" "drop_layer" "JS Div(mean)" "JS Div(std)" "Perception" "Cognition"
printf "%s\n" "-------------|----------------|------------|----------------|---------------"

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
    # JS divergence
    js_mean = js_data.get(dl, {}).get('mean', float('nan'))
    js_std  = js_data.get(dl, {}).get('std', float('nan'))

    # MME scores
    perc, cogn = '-', '-'
    score_file = '${RESULT_DIR}/mme_score_drop' + dl + '.txt'
    if os.path.exists(score_file):
        with open(score_file) as sf:
            for line in sf:
                m = re.match(r'^Perception\D*([\d.]+)', line)
                if m: perc = m.group(1)
                m = re.match(r'^Cognition\D*([\d.]+)', line)
                if m: cogn = m.group(1)

    print(f'{dl:<12} | {js_mean:<14.6f} | {js_std:<10.6f} | {perc:<14} | {cogn:<14}')
"

echo ""
echo "========== All done! =========="
} 2>&1 | tee $SUMMARY_FILE
