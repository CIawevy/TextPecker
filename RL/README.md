Our RL framework is built upon [Flow-GRPO](https://github.com/yifan123/flow_grpo). We provide training code for optimizing text rendering models with **TextPecker Reward**.

## 1. Environment Setup

```bash
cd RL/flow_grpo
conda create -n flow_grpo python=3.10.16 -y
conda activate flow_grpo
pip install -e .
pip install git+https://github.com/huggingface/diffusers.git
pip install -r requirements.txt
pip install flash_attn==2.6.1 --no-build-isolation 
cd ..
```

## 2. Multi-Reward Setup

To configure the multi-reward environment, refer to [**Flow-GRPO**](https://github.com/yifan123/flow_grpo) and download the required model weights. You may also use the script `../scripts/download_model.sh` to fetch the weights automatically.

### Verify TextPecker
Prior to initiating training, run the following code to verify that the TextPecker service can run properly:
```bash
python ../eval/TextPecker_eval/demo.py
```

## 3. Training
We incorporate recent GRPO training techniques to improve efficiency and stability, including FlowGRPO-Fast (or [MixGRPO](https://arxiv.org/abs/2507.21802)), [GRPO-Guard](https://arxiv.org/abs/2510.22319), velocity KL loss, etc.

We provide RL training code and backbone-specific launch scripts under `flow_grpo/scripts/multi_node/text_rendering`, covering:
- SD3.5-M:`flow_grpo/config/SD.py`
- FLUX.1-dev:`flow_grpo/config/FLUX.py`
- QwenImage:`flow_grpo/config/QWEN.py`

Example: 
```bash
bash flow_grpo/scripts/multi_node/text_rendering/sd3_grpofastguard_vkl_pecker.sh
```
Datasets:
We also provide more challenging bilingual (Chinese & English) text-rendering datasets with richer text-length distributions:

- `flow_grpo/dataset/ocr_v2` (Chinese & English prompts)
- `flow_grpo/dataset/ocr_v2_en` (English prompts)