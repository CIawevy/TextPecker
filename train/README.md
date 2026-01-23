**TextPecker training, deployment, and evaluation are built on top of [ms-swift](https://github.com/modelscope/ms-swift).

We provide two model checkpoints:
- [TextPecker-8B-Qwen3VL](https://huggingface.co/ByteDance/TextPecker-8B-Qwen3VL)
- [TextPecker-8B-InternVL3](https://huggingface.co/ByteDance/TextPecker-8B-InternVL3_5)

Here we provide example scripts for TextPecker training, deployment, and evaluation. You can refer to the [Supported Models and Datasets](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html) for more help.

## 1. Environment Setup
```bash
git clone https://github.com/bytedance/TextPecker.git
cd TextPecker/train
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
conda create -n TextPecker python=3.11.13 -y
conda activate TextPecker
pip install -e .
cd ..
sh install_all.sh
```

## 2. Training and Evaluation

We provide backbone-specific scripts under the `train/` folder:
- Qwen3VL: `train_textpecker_qwen3vl.sh`, `deploy_server_qwen3vl.sh`, `qwenvl_infer.sh`
- InternVL3: `train_textpecker_internvl3.sh`, `deploy_server_internvl.sh`, `internvl_infer.sh`

Before Training, run our demo code to verify TextPecker reward
Example:
```bash
bash train_textpecker_internvl3.sh
```

After training, you can evaluate the model with:
```bash
bash internvl_infer.sh
```

Convert outputs and compute metrics:
```bash
conda activate TextPecker
python convert_api_infer_results_swift.py <your_output_file>
python get_metrics_for_tsap_and_ctr.py --language <chinese|english|all> 
```

## 3. Deployment

We provide scripts for launching serving endpoints. Example:
```bash
bash deploy_server_internvl.sh <GPU_MEMORY_UTILIZATION> <PORT> <CUDA_VISIBLE_DEVICES>