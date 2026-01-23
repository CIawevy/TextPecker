
source your_conda_env/bin/activate #replace with your own
conda activate TextPecker
# export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=your_cache_dir #replace with your own
export HF_HOME=your_cache_dir #replace with your own
cd TextPecker/train/ms-swift #replace with your own

#load data online
DATA="ByteDance/TextPecker-1.5M"
CUSTOM_DATASET_INFO='[
  {
    "hf_dataset_id": "ByteDance/TextPecker-1.5M",
    "split": ["test"],
    "subsets": ["default"]
  }
]'
# load data offline
# DATA=your_data_path
# CUSTOM_DATASET_INFO='[
#   {
#     "dataset_path": "your_data_path",
#     "split": ["test"],
#     "subsets": ["default"]
#   }
# ]'

OUTPUT_FILE=inference-output/eval_results_it.jsonl #replace with your own

MODEL=ByteDance/TextPecker-8B-InternVL3

CUDA_VISIBLE_DEVICES=0,1,2,3\
MAX_NUM=12 \
INPUT_SIZE=448 \
swift infer \
    --model $MODEL\
    --stream true \
    --infer_backend pt \
    --val_dataset $DATA\
    --max_new_tokens 2048 \
    --result_path $OUTPUT_FILE \
    --use_hf True \
    --custom_dataset_info "$CUSTOM_DATASET_INFO" \
    --remove_unused_columns false

