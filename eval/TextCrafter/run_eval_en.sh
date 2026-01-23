#!/bin/bash
source your_conda_env/bin/activate #replace with your own
conda activate textcrafter_eval

# 进入工作目录
cd TextPecker/eval/TextCrafter/TextCrafter_Eval

# 定义基础路径
BASE_FOLDER=../CVTG-2K/eval_results
BENCHMARK_DIR=../CVTG-2K

# 定义模型名称列表
MODEL_NAMES=(
   "MODEL_NAME_0" "MODEL_NAME_1"
)

# 遍历模型名称列表
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Processing model: $MODEL_NAME"
    
    # 设置输出目录和结果目录
    OUTPUT_DIR=./eval_metrics/${MODEL_NAME}
    RESULT_DIR=${BASE_FOLDER}/${MODEL_NAME}
    
    echo "RESULT_DIR: $RESULT_DIR"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    
    # 检查结果目录是否存在
    if [ ! -d "$RESULT_DIR" ]; then
        echo "Error: Result directory $RESULT_DIR does not exist. Skipping..."
        continue
    fi
    
    # 创建输出目录（如果不存在）
    mkdir -p "$OUTPUT_DIR"

    
    # 执行评估
    python unified_metrics_eval.py --benchmark_dir "$BENCHMARK_DIR" --result_dir "$RESULT_DIR" --output_file "$OUTPUT_DIR/results.json" --device auto
    
    # 检查评估是否成功
    if [ $? -ne 0 ]; then
        echo "Error: Evaluation for model $MODEL_NAME failed."
    else
        echo "Successfully completed evaluation for model $MODEL_NAME"
    fi
done

# 退出conda环境
conda deactivate