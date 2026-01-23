#!/bin/bash

# 设置基本参数
base_port=8848
gpu_memory_utilization=0.85  #replace with your own
script_path=train/deploy_server_qwen3vl.sh #replace with your own
# 定义带时间戳的日志目录
timestamp=$(date +'%Y%m%d_%H%M%S')
log_dir=ms-swift/server_logs_$timestamp #replace with your own 
mkdir -p "$log_dir"

# 启动四组服务器，每组使用不同的GPU和端口 #replace with your own
echo "Starting server group 0-1..."
bash $script_path $gpu_memory_utilization $base_port "0,1" > "$log_dir/server_01.log" 2>&1 &
echo "Server group 0-1 started on port $base_port"

echo "Starting server group 2-3..."
bash $script_path $gpu_memory_utilization $((base_port+1)) "2,3" > "$log_dir/server_23.log" 2>&1 &
echo "Server group 2-3 started on port $((base_port+1))"

echo "Starting server group 4-5..."
bash $script_path $gpu_memory_utilization $((base_port+2)) "4,5" > "$log_dir/server_45.log" 2>&1 &
echo "Server group 4-5 started on port $((base_port+2))"

echo "Starting server group 6-7..."
bash $script_path $gpu_memory_utilization $((base_port+3)) "6,7" > "$log_dir/server_67.log" 2>&1 &
echo "Server group 6-7 started on port $((base_port+3))"

echo "All servers started. Check logs in $log_dir for details."

