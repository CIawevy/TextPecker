
#!/bin/bash
# 包含所有 VLLM 服务器地址、本地地址及内部域名后缀
# export NO_PROXY="2605:340:cd60:0:9e1c:a6fd:1ee2:d01b"
# Common part for all nodes
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5
# export NCCL_DEBUG=WARN
# export NCCL_IB_GID_INDEX=3
MASTER_ADDR=${MASTER_ADDR:?need master addr}
MASTER_PORT=${MASTER_PORT:-35201}
NNODES=${NNODES:-4}
NODE_RANK=${NODE_RANK:?need node_rank}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}


export NCCL_TIMEOUT=3600000




source your_conda_env/bin/activate 
conda activate flow_grpo
cd TextPecker/RL/flow_grpo
# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=10.82.139.22
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file  scripts/accelerate_configs/deepspeed_zero2.yaml\
    --num_machines ${NNODES} --num_processes ${NPROC_PER_NODE} \
    --machine_rank ${NODE_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_flux_fast_guard.py \
    --config config/FLUX.py:general_ocr_flux_fast_guard_vkl_pecker_mrtpa &



wait
echo "所有进程完成"