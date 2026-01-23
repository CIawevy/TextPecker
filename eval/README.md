**We provide re-evaluation guidelines with TextPecker for the following benchmarks: OneIG, CVTG, LongText, TextAtlasEval, LeX-Bench, and TIIF-Bench.**

## 1️⃣ Environment Setup

First, set up the evaluation environment with the following steps:

```bash
cd eval
conda create -n TextPeckerEval python=3.10.16 -y
conda activate TextPeckerEval
pip install -r framework.txt
sh install_all.sh
```

## 2️⃣ Model Inference

We provide benchmark-specific inference code and backbone-specific launch scripts under `eval/models`, supporting:
- SD3.5-M: `eval/models/SD3.5/run_scripts.sh`
- FLUX.1-dev: `eval/models/Flux/run_scripts.sh`
- QwenImage: `eval/models/Qwenimage/run_scripts.sh`

## 3️⃣ Verify TextPecker Server

Before proceeding with evaluation, verify that the TextPecker server is running as expected:

```bash
python TextPecker_eval/demo.py
```

## 4️⃣ Evaluate with TextPecker

Run the following commands to perform evaluation and calculate metrics:

```bash
cd TextPecker_eval
bash eval_universal_bench.sh  
bash cal_metrics.sh           
```