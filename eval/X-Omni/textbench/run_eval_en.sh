#!/bin/bash
# 激活conda环境
source source your_conda_env/bin/activate #replace with your own
conda activate TextPecker

# 进入工作目录
cd TextPecker/eval/X-Omni/textbench

# 定义基础路径和模式
BASE_FOLDER=../eval_results
MODE=en # en or zh

# 定义模型名称列表
MODEL_NAMES=(
    "MODEL_NAME_0"
    "MODEL_NAME_1"
    )


# 遍历模型名称列表
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Processing model: $MODEL_NAME"
    
    # 设置输出目录
    OUTPUT_DIR=eval_metrics/${MODEL_NAME}
    SAMPLE_FOLDER=${BASE_FOLDER}/${MODEL_NAME}
    echo "SAMPLE_FOLDER: $SAMPLE_FOLDER"
    # 创建输出目录（如果不存在）
    mkdir -p "$OUTPUT_DIR"
    FREE_PORT=$(bash get_port.sh)
    # 执行评估
    torchrun --nnodes=1 --node-rank=0 --nproc_per_node=8 --master-port="$FREE_PORT"\
        evaluate_text_reward.py \
        --sample_dir "$SAMPLE_FOLDER" \
        --output_dir "$OUTPUT_DIR" \
        --mode "$MODE"
    
    # 合并结果文件
    if ls $OUTPUT_DIR/results_chunk*.jsonl 1> /dev/null 2>&1; then
        cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results.jsonl
        rm $OUTPUT_DIR/results_chunk*.jsonl
        echo "Successfully merge chunks！"
    else
        echo "Warning: No results_chunk files found for $OUTPUT_DIR"
    fi
    
    # 生成摘要分数
    if [ -f "$OUTPUT_DIR/results.jsonl" ]; then
        python3 summary_scores.py "$OUTPUT_DIR/results.jsonl" --mode "$MODE"
    else
        echo "Warning: results.jsonl not found for $MODEL_NAME"
    fi
    
    echo "Completed processing for $MODEL_NAME"
done

# 退出conda环境（可选）
conda deactivate