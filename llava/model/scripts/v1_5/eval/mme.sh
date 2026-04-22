# 定义模型
MODEL=llava-v1.5-7b
#MODEL=llava-v1.5-13b
MODEL2=llava-v1.6-vicuna-7b
# 执行评估脚本
python -m llava.eval.model_vqa_loader \
    --model-path liuhaotian/$MODEL \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --visual_token_num 576 \
    --layer_list       '[1,10,15]'\
    --image_token_list '[218, 128, 54]'\
    --conv-mode vicuna_v1 \



# 进入工作目录
cd ./playground/data/eval/MME

# 转换答案文件
python convert_answer_to_mme.py --experiment $MODEL

# 切换到评估工具目录
cd eval_tool

# 计算评估结果
python calculation.py --results_dir answers/$MODEL
