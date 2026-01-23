# We provide scripts for launching serving endpoints. Example:
# bash deploy_server_version.sh <GPU_MEMORY_UTILIZATION> <PORT> <CUDA_VISIBLE_DEVICES>

#deploy TextPecker-8B-InternVL3_5
bash deploy_server_internvl.sh 0.85 8848 0,1
#deploy TextPecker-8B-Qwen3VL
bash deploy_server_qwen3vl.sh 0.85 8849 2,3
