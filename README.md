<p align="center">
  <img src="assets/logo.png" alt="TextPecker" width="480"/>
</p>

<p align="center">
  <a href="https://github.com/CIawevy/TextPecker">
    <img
      src="https://img.shields.io/badge/TextPecker-Website-0A66C2?logo=safari&logoColor=white"
      alt="TextPecker Website"
    />
  </a>
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/TextPecker-Paper-red?logo=arxiv&logoColor=red"
      alt="TextPecker Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/CIawevy/TextPecker-8B-InternVL3">
    <img 
        src="https://img.shields.io/badge/TextPecker-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="TextPecker Model"
    />
  </a>
  <a href="https://github.com/CIawevy/TextPecker/blob/main/eval/TextPecker_eval/demo.py">
    <img
      src="https://img.shields.io/badge/TextPecker-Demo-blue?logo=googleplay&logoColor=blue"
      alt="TextPecker Demo"
    />
  </a>
   <a href="https://huggingface.co/datasets/CIawevy/TextPecker-1.5M">
    <img 
        src="https://img.shields.io/badge/TextPecker1.5M-Dataset-orange?logo=huggingface&logoColor=yellow" 
        alt="TextPecker-1.5M Dataset"
    />
  </a>
  <!-- <a href="https://huggingface.co/spaces/ByteDance/TextPecker">
    <img 
        src="https://img.shields.io/badge/TextPecker-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="TextPecker Model"
    />
  </a> -->
  <!-- <a href="https://discord.gg/eXQNFhWe">
    <img
      src="https://img.shields.io/badge/TextPecker-Discord-5865F2?logo=discord&logoColor=purple"
      alt="TextPecker Discord"
    />
  </a> -->
  <!-- <a href="mailto:TextPecker@bytedance.com">
    <img
      src="https://img.shields.io/badge/TextPecker-Email-D14836?logo=gmail&logoColor=red"
      alt="TextPecker Email"
    />
  </a>
</p> -->

# TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering

> [Hanshen Zhu](https://scholar.google.com/citations?user=1tjNZlcAAAAJ&hl=en), [Yuliang Liu](https://scholar.google.com/citations?user=9uPDtI4AAAAJ&hl=zh-CN&authuser=1), [Xuecheng Wu](https://scholar.google.com/citations?user=MuTEp7sAAAAJ&hl=zh-CN), [An-Lan Wang](https://scholar.google.com/citations?user=mazWHncAAAAJ&hl=en), [Hao Feng](https://scholar.google.com/citations?user=aB8DspEAAAAJ&hl=zh-CN), [Dingkang Yang](https://scholar.google.com/citations?user=jvlDhkcAAAAJ&hl=zh-CN), [Chao Feng](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AIfU4H7WFVeWuU5GeELC6VB1oKluoC7ZQlQxUCGcDWqHhdX2iIfDMrfcw1Jjj3dUejmTU2gS8q8ey1quzooAZ0VLTTBW4O5iVcfeqpf_7HU&user=4eEryIsAAAAJ), [Can Huang](https://scholar.google.com/citations?user=ON9Rx-IAAAAJ&hl=zh-CN), [Jingqun Tang](https://scholar.google.com/citations?user=OxQXSioAAAAJ&hl=zh-CN)<sup>†</sup>, [Xiang Bai](https://scholar.google.com/citations?user=UeltiQ4AAAAJ&hl=en)<sup>†</sup>
> ## Abstract
>Visual Text Rendering (VTR) remains a critical challenge in text‑to‑image generation, where even advanced models frequently produce text with structural anomalies such as distortion, blurriness, and misalignment.
However, we find that leading MLLMs and specialist OCR models largely fail to perceive these structural anomalies, creating a critical bottleneck for both VTR evaluation and RL‑based optimization.   
As a result, even state‑of‑the‑art generators (e.g., SeedDream4.0, Qwen‑Image) still struggle to render structurally faithful text.
To address this, we propose **TextPecker**,
a plug-and-play structural anomaly perceptive RL strategy that mitigates noisy reward signals and works with any text-to-image generator. 
To enable this capability, we construct a recognition dataset with character‑level structural‑anomaly annotations and develop a stroke‑editing synthesis engine to expand structural‑error coverage. 
Experiments show that TextPecker consistently improves diverse text‑to‑image models; even on the well‑optimized Qwen‑Image, it significantly yields average gains of 4% in structural fidelity and 8.7% in semantic alignment for Chinese text rendering, establishing a new state-of-the-art in high-fidelity VTR.
Our work fills a gap in VTR optimization, providing a foundational step towards  reliable and structural faithful visual text generation.

<p align="center"><img src="assets/method.png" width="95%"></p>
<!-- <p align="center"><img src="assets/motivation.png" width="95%"></p> -->
<!-- <p align="center"><img src="assets/data_pipe.png" width="95%"></p> -->
<!-- <p align="center"><img src="assets/eval.png" width="95%"></p> -->

[//]: # (This repository represents the official implementation of the paper titled "TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering".)

## 📢 News
<!-- - **Feb 22, 2026:** We released optimized model checkpoints: [Qwen Image](), [SD3.5-M](), and [Flux.1-[dev]]().
- **Feb 20, 2026:** Our Arxiv Paper is now publicly available.
- **Feb 20, 2026:**  TextPecker has been accepted to **CVPR 2026**. -->
- **Feb 18, 2026:** We released the LoRA weights for different TextPecker-optimized generative models, including:
[SD3.5-M](https://huggingface.co/CIawevy/SD3.5M-TextPecker-SQPA), [Flux.1-dev](https://huggingface.co/CIawevy/Flux.1-dev-TextPecker-SQPA), [Qwen-Image](https://huggingface.co/CIawevy/QwenImage-TextPecker-SQPA). 
- **Feb 15, 2026:** We released the official [website](https://github.com/CIawevy/TextPecker),[model](https://huggingface.co/CIawevy/TextPecker-8B-InternVL3), [dataset](https://huggingface.co/datasets/CIawevy/TextPecker-1.5M) for TextPecker.



## 🔥 Quick Start

Training, deployment, and evaluation of TextPecker are all built upon [ms-swift](https://github.com/modelscope/ms-swift). We currently provide two versions of model checkpoints: [TextPecker-8B-Qwen3VL](https://huggingface.co/CIawevy/TextPecker-8B-Qwen3VL) and [TextPecker-8B-InternVL3](https://huggingface.co/CIawevy/TextPecker-8B-InternVL3). For detailed environment setup and model deployment/testing instructions, please refer to the [official documentation](https://swift.readthedocs.io/en/latest/index.html). 

1️⃣ Environment Setup
```bash
git clone https://github.com/CIawevy/TextPecker.git
cd TextPecker/train
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
conda create -n TextPecker python=3.11.13 -y
conda activate TextPecker
pip install -e .
cd ..
sh install_all.sh
```

2️⃣ Download Models & Dataset

We have uploaded our models and datasets to Hugging Face. You can download them using the provided scripts. Modify parameters (e.g., local paths, HF token) in `scripts/download_models.sh` and `scripts/download_dataset.sh` as needed, then run `bash scripts/download_xxx.sh` (for models / datasets). Additionally, refer to [DATA](https://github.com/CIawevy/TextPecker_engine) to use our data engine for synthesizing your own datasets if needed.

3️⃣ Deployment （See [TRAIN](train/) for more details.）

Example

```bash
bash train/deploy_textpecker.sh
```
4️⃣ Demo

After deployment, you can run the following command to try our demo:
```bash
python eval/TextPecker_eval/demo.py
```


## 🔥 Train & Eval

### TextPecker training

TextPecker training, deployment, and evaluation are built on top of [ms-swift](https://github.com/modelscope/ms-swift). We provide backbone-specific training scripts under `train` folder. See [TRAIN](train) for more details.

  
### VTR RL Training (TextPecker Reward)

Our RL framework builds on [Flow-GRPO](https://github.com/yifan123/flow_grpo). We provide training code for optimizing text rendering models with TextPecker under `./RL/flow_grpo/`. For details, please refer to [RL](RL).


### Re-evaluate Benchmarks with TextPecker


TextPecker can evaluate text structural quality and image-level or box-level semantic consistency for any text generation or editing scenarios. We provide re-evaluation instructions for the following benchmarks: OneIG-Bench, CVTG-2K, LongText, TextAtlas, LeX-Bench, and TIIF-Bench. For more details, see [EVAL](eval).

## 🤗 Resources
We fully open-source all core resources of the TextPecker ecosystem, including evaluators, Optimized VTR models, and datasets, to facilitate research and application development.
## Evaluator
| Variant   | Model |
| --------- | ----- |
| InternVL-3 | [TextPecker-8B-InternVL3](https://huggingface.co/CIawevy/TextPecker-8B-InternVL3) |
| Qwen3-VL   | [TextPecker-8B-Qwen3VL](https://huggingface.co/CIawevy/TextPecker-8B-Qwen3VL) |

## VTR Models
| Variant     | Model |
| ----------- | ----- |
| SD3.5-M     | [SD3.5M-TextPecker-SQPA](https://huggingface.co/CIawevy/SD3.5M-TextPecker-SQPA) |
| Flux.1-dev  | [Flux.1-dev-TextPecker-SQPA](https://huggingface.co/CIawevy/Flux.1-dev-TextPecker-SQPA) |
| Qwen-Image  | [QwenImage-TextPecker-SQPA](https://huggingface.co/CIawevy/QwenImage-TextPecker-SQPA) |

### Dataset & Engine
| Type               | Link |
| ------------------ | ---- |
| Evaluator Dataset  | [TextPecker-1.5M](https://huggingface.co/datasets/CIawevy/TextPecker-1.5M) |
| VTR RL Dataset | [TextPecker-RL](RL/flow_grpo/dataset) |
| Engine   | [TextPecker-engine](https://github.com/CIawevy/TextPecker_engine) |

# Acknowledgement
We sincerely thank 
[ms-swift](https://github.com/modelscope/ms-swift#), 
[Flow-GRPO](https://github.com/yifan123/flow_grpo) 
for their valuable methodological contributions.

Additionally, we appreciate the support of 
[TextAtlas5M](https://github.com/CSU-JPG/TextAtlas), 
[LeX-10k](https://github.com/zhaoshitian/LeX-Art), 
[SynTIGER](https://github.com/clovaai/synthtiger),
[WanJuan1.0](https://github.com/opendatalab/WanJuan1.0), 
[Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev),
[Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), 
[SD3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), 
[CogView4](https://github.com/zai-org/CogView4), 
[Kolors](https://github.com/Kwai-Kolors/Kolors) and 
[Seedream4.0](https://seed.bytedance.com/en/seedream4_0)
for their role in data generation. 

We also thank the evaluation benchmarks including 
[CVTG-2K](https://github.com/NJU-PCALab/TextCrafter), 
[LongText](https://github.com/X-Omni-Team/X-Omni), 
[OneIG-Bench](https://github.com/OneIG-Bench/OneIG-Benchmark), 
[TIIF-Bench](https://github.com/A113N-W3I/TIIF-Bench), 
[TextAtlas](https://github.com/CSU-JPG/TextAtlas) and 
[LeX-Bench](https://github.com/zhaoshitian/LeX-Art) for facilitating text rendering evaluation.

## ✍️ Citation
If you find TextPecker useful in your research or work, please cite our paper:
```bibtex
@article{zhu2026TextPecker,
  title   = {TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering},
  author  = {Zhu, Hanshen and Liu, Yuliang and Wu, Xuecheng and Wang, An-Lan and Feng, Hao and Yang, Dingkang and Feng, Chao and Huang, Can and Tang, Jingqun and Bai, Xiang},
  journal = {arXiv preprint arXiv:},
  year    = {2026}
}
```


## 📜 License
TextPecker is licensed under the Apache 2.0.