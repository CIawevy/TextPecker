#!/bin/bash
# TextPecker 通用评测脚本 - 支持多个数据集

# 激活conda环境（请根据实际情况修改路径）
source your_conda_env_path/etc/profile.d/conda.sh
conda activate TextPeckerEval

# 进入工作目录
cd TextPecker/eval #replace with your own

# --- 配置部分 --- #

# 模型列表 (格式: <MODEL_NAME>_<LANGUAGE>)
# 示例: MODEL_NAMES=("Flux_EN" "SD3.5_EN" "QwenImage_ZH" "QwenImage_EN")
MODEL_NAMES=(
    "Flux" #without suffix
    "SD3.5" #without suffix
    "QwenImage" #without suffix
)

# 数据集配置
# 格式: "<数据集类型>:<基础图像路径>:<输出结果路径>:<mode>"
# 支持的数据集类型: longtext, cvtg, oneig, gentexteval
# 支持的mode: auto, en, zh
#  IFS=":" read -r DATASET_TYPE BASE_FOLDER OUTPUT_BASE DATASET_MODE <<< "$dataset"
DATASETS=(
    "longtext:X-Omni/eval_results:TextPecker_eval/results/longtext:auto"
    "cvtg:TextCrafter/CVTG-2K/eval_results:TextPecker_eval/results/cvtg:en"
    "oneig:OneIG-Benchmark/images/text:TextPecker_eval/results/oneig:auto"
    "gentexteval:GenTextEval/eval_results:TextPecker_eval/results/gentexteval:auto"
    "lex:LeX-Bench/eval_results:TextPecker_eval/results/lex:en"
    "atlas:TextAtlasEval/eval_results:TextPecker_eval/results/atlas:en"
    "tiif:TIIF-Bench/eval_results:TextPecker_eval/results/tiff:en"


)

# VLLM 端口配置
PORT=8848 
VLLM_HOST="2605:340:cd60:0:90cb:803d:7dc0:99a8" # replace with your HOST
# --- 执行部分 --- #

# 检查模型列表是否为空
if [ ${#MODEL_NAMES[@]} -eq 0 ]; then
    echo "错误: 请在 MODEL_NAMES 中指定要评测的模型"
    exit 1
fi

# 遍历所有数据集
for dataset in "${DATASETS[@]}"; do
    # 解析数据集配置
    IFS=":" read -r DATASET_TYPE BASE_FOLDER OUTPUT_BASE DATASET_MODE <<< "$dataset"
    
    echo "\n=== 开始评测数据集: $DATASET_TYPE ==="
    echo "基础图像路径: $BASE_FOLDER"
    echo "结果输出基础路径: $OUTPUT_BASE"
    echo "数据集模式: $DATASET_MODE"
    
    # 遍历所有模型
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        echo "\n处理模型: $MODEL_NAME"
        
        # 设置样本目录和输出目录
        SAMPLE_FOLDER="$BASE_FOLDER/$PROCESS_MODEL_NAME"
        OUTPUT_DIR="$OUTPUT_BASE/$PROCESS_MODEL_NAME"
        
        echo "样本目录: $SAMPLE_FOLDER"
        echo "输出目录: $OUTPUT_DIR"
        
        # 检查样本目录是否存在
        # if [ ! -d "$SAMPLE_FOLDER" ]; then
        #     echo "警告: 样本目录 $SAMPLE_FOLDER 不存在，跳过该模型"
        #     continue
        # fi
        
        # 创建输出目录
        mkdir -p "$OUTPUT_DIR"
        
        # 执行评测
        python TextPecker_eval/textpecker_eval_server.py \
            --sample_dir "$SAMPLE_FOLDER" \
            --output_dir "$OUTPUT_DIR" \
            --mode "$DATASET_MODE" \
            --type "$DATASET_TYPE" \
            --port "$PORT" \
            --vllm_host "$VLLM_HOST" \
        
        echo "模型 $PROCESS_MODEL_NAME 的 $DATASET_TYPE 数据集评测完成"
    done
    
    echo "=== 数据集 $DATASET_TYPE 评测完成 ==="
done

echo "\n✅ 所有数据集评测完成！"