#!/bin/bash

# 1. 首先从参数中提取值
GPU_MEM_UTILIZATION=${1:-0.9}
FREE_PORT=${2:-8849}
GPUS=${3:-4,5}

# 2. 然后激活环境（此时不会错误地传递参数）
source your_conda_env/bin/activate #replace with your own
conda activate TextPecker

export MODELSCOPE_CACHE=your_cache_dir #replace with your own
cd TextPecker/train/ms-swift #replace with your own


CUDA_VISIBLE_DEVICES=$GPUS \
MAX_NUM=12 \
INPUT_SIZE=448 \
swift deploy \
    --model ByteDance/TextPecker-8B-InternVL3 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization $GPU_MEM_UTILIZATION \
    --vllm_tensor_parallel_size 2 \
    --vllm_max_model_len 32768 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 1}' \
    --served_model_name TextPecker \
    --host :: \
    --vllm_use_async_engine true \
    --port $FREE_PORT


    