#!/bin/bash
# TextPecker 通用评测脚本 - 支持多个数据集

# 激活conda环境（请根据实际情况修改路径）
source your_conda_env_path/etc/profile.d/conda.sh
conda activate TextPeckerEval

# 进入工作目录
cd TextPecker/eval/TextPecker_eval #replace with your own
RESULT_DIR=results #path of eval_results
OUTPUT_FILE=metric_summary.txt #path to save metrics_summary

python metric_summary.py --result_dir $RESULT_DIR --output_file $OUTPUT_FILE

