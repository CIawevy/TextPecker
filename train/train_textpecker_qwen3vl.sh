source your_conda_env/bin/activate #replace with your own
conda activate TextPecker
export MODELSCOPE_CACHE=your_cache_dir #replace with your own
cd TextPecker/train/ms-swift #replace with your own

MASTER_ADDR=${MASTER_ADDR:?need master addr}
MASTER_PORT=${MASTER_PORT:-29501}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:?need node_rank}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

OUTPUT_DIR=TextPecker-InterlVL3_5-8B
DATA="ByteDance/TextPecker-1.5M"
CUSTOM_DATASET_INFO='[
  {
    "hf_dataset_id": "ByteDance/TextPecker-1.5M",
    "split": ["train"],
    "subsets": ["default"]
  }
]'
BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    swift/cli/sft.py \
    --model $BASE_MODEL \
    --dataset $DATA \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps $(expr 32 / $NPROC_PER_NODE / $NNODES) \
    --system 'You are a helpful assistant.' \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 100 \
    --max_length 8192 \
    --output_dir $OUTPUT_DIR \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --use_hf True \
    --custom_dataset_info "$CUSTOM_DATASET_INFO" \
    --save_only_model true