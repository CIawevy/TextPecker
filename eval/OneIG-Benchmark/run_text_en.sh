#!/bin/bash
source your_conda_env/bin/activate #replace with your own
conda activate oneIG
cd  TextPecker/eval/OneIG-Benchmark
# start_time
start_time=$(date +%s)

# mode (EN/ZH)
MODE=EN #ZH

# image_root_dir
IMAGE_DIR="images/text"

# model list
BASE_MODEL_NAMES=( 
  "MODEL_NAME_0"
  "MODEL_NAME_1"
)

 
MODEL_NAMES=()
for model in "${BASE_MODEL_NAMES[@]}"; do
  MODEL_NAMES+=("${model}_${MODE}")  # 结果如 "SD3.5_EN"
done


# image grid
IMAGE_GRID=()
for ((i=0; i<${#BASE_MODEL_NAMES[@]}; i++)); do
  IMAGE_GRID+=(2)  # 为每个模型设置网格大小为2
done


# Text Score

echo "It's text time."

python -m scripts.text.text_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" 

rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."