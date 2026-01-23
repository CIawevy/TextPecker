---
license: cc-by-nc-4.0
configs:
  - config_name: OneIG-Bench
    data_files: OneIG-Bench.json
  - config_name: OneIG-Bench-ZH
    data_files: OneIG-Bench-ZH.json
task_categories:
  - text-to-image
---


<div style="text-align: center;">
    <h1>OneIG-Bench</h1>
    <h2>Omni-dimensional Nuanced Evaluation for Image Generation</h2>
    <div class="is-size-4 publication-authors" align="center">
        <span class="author-block">
            <a href="https://scholar.google.com/citations?user=i1accPcAAAAJ&hl=en">Jingjing Chang</a><sup>1,2</sup>,
        </span>
        <span class="author-block">
            <a href="https://scholar.google.com/citations?user=yDNIBE0AAAAJ&hl=en">Yixiao Fang</a><sup>2&dagger;</sup>,
        </span>
        <span class="author-block">
            <a href="https://scholar.google.com/citations?user=k_jXsNYAAAAJ&hl=en">Peng Xing</a><sup>2</sup>,
        </span>
        <span class="author-block">
            <a href="https://openreview.net/profile?id=~Shuhan_Wu3">Shuhan Wu</a><sup>2</sup>,
        </span>
        <span class="author-block">
            <a href="https://wchengad.github.io">Wei Cheng</a><sup>2</sup>,
        </span>
    </div>
    <div class="is-size-4 publication-authors" align="center">
        <span class="author-block">
            <a href="https://wrong.wang/">Rui Wang</a><sup>2</sup>,
        </span>
        <span class="author-block">
            <a href="https://scholar.google.com/citations?user=tgDc0fsAAAAJ&hl=en">Xianfang Zeng</a><sup>2</sup>,
        </span>
        <span class="author-block">
            <a href="https://www.skicyyu.org/">Gang Yu</a><sup>2&ddagger;</sup>,
        </span>
        <span class="author-block">
            <a href="https://dmne.sjtu.edu.cn/dmne/faculty/chenhaibao/">Hai-Bao Chen</a><sup>1&ddagger;</sup>
        </span>
    </div>
    <div class="is-size-5 publication-authors" align="center">
        <span class="author-block"><sup>&dagger;</sup>Project lead,</span>
        <span class="author-block"><sup>&ddagger;</sup>Correspondence Authors</span>
    </div>
    <div class="is-size-5 publication-authors" align="center">
        <span class="author-block"><sup>1</sup>Shanghai Jiao Tong University,</span>
        <span class="author-block"><sup>2</sup>StepFun</span>
    </div>
</div>
<div style="text-align: center;">
    <br>
    <a href="https://oneig-bench.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
    <a href="https://github.com/OneIG-Bench/OneIG-Benchmark"><img src="https://img.shields.io/badge/GitHub-Code-blue?logo=github"></a> &ensp;
    <a href="https://arxiv.org/abs/2506.07977"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
    <br>
    <img src="./assets/OneIG-Bench.png">
</div>



## 🔥🔥🔥 News

* **`2025.06.10`** 🌟 We released the [`OneIG-Bench`](https://huggingface.co/datasets/OneIG-Bench/OneIG-Bench) benchmark on 🤗huggingface.
* **`2025.06.10`** 🌟 We released the [tech report](https://arxiv.org/abs/2506.07977) and the [project page](https://oneig-bench.github.io/)
* **`2025.06.10`** 🌟 We released the evaluation scripts. 

## To Do List
- [ ] Real-time Updating Leaderboard
- [x] OneIG-Bench Release 
- [x] Evaluation Scripts, Technical Report & Project Page Release

## Introduction

We introduce OneIG-Bench, a meticulously designed comprehensive benchmark framework for fine-grained evaluation of T2I models across multiple dimensions, including subject-element alignment, text rendering precision, reasoning-generated content, stylization, and diversity. Specifically, these dimensions can be flexibly selected for evaluation based on specific needs.

Key contribution:

- We present **OneIG-Bench**, which consists of six prompt sets, with the first five — 245 *Anime and Stylization*, 244 *Portrait*, 206 *General Object*, 200 *Text Rendering*, and 225 *Knowledge and Reasoning* prompts — each provided in both English and Chinese, and 200 *Multilingualism* prompts, designed for the comprehensive evaluation of current text-to-image models.
- A systematic quantitative evaluation is developed to facilitate objective capability ranking through standardized metrics, enabling direct comparability across models. Specifically, our evaluation framework allows T2I models to generate images only for prompts associated with a particular evaluation dimension, and to assess performance accordingly within that dimension.
- State-of-the-art open-sourced methods as well as the proprietary model are evaluated based on our proposed benchmark to facilitate the development of text-to-image research.

## Dataset

Our benchmark includes **OneIG-Bench** and **OneIG-Bench-ZH**. The format of **OneIG-Bench** is as follows:
```bash
{
    "category": "Anime_Stylization",     # Category of the prompt
    "id": "095",                         # Unique identifier
    "prompt_en": "{prompt}",             # English prompt
    "type": "NP",                        # Type of prompt (e.g., NP or T,P)
    "prompt_length": "long",             # Length of the prompt (e.g., short, middle, long)
    "class": "impasto"                   # Style in Anime & Stylization, Form in Text Rendering, Subject in Knowledge & Reasoning 
},
...
```
**OneIG-Bench-ZH** is a simplified version of OneIG-Bench, where the 'type' and 'prompt_length' fields are omitted. After generating the images, a fine-grained analysis can be conducted based on the 'type', 'prompt_length', and 'class' fields.

## Get Started

### Image Generation
It's better for you to generate 4 images for each prompt in OneIG-Bench. And combine these 4 images into a single image. Each prompt's generated images should be saved into subfolders based on their category **Anime & Stylization, Portrait, General Object, Text Rendering, Knowleddge & Reasoning, Multilingualism**, corresponding to folders **anime, human, object, text, reasoning, multilingualism**. If any image cannot be generated, I suggest saving a black image with the specified filename.

The filename for each image should follow the id assigned to that prompt in [`OneIG-Bench.json`](OneIG-Bench.json)/[`OneIG-Bench-ZH.json`](OneIG-Bench-ZH.json). The structure of the images to be saved should look like:

```bash
📁 images/
├── 📂 anime/                  
│   ├── 📂 gpt-4o/
│   │   ├── 000.webp
│   │   ├── 001.webp
│   │   └── ...
│   ├── 📂 imagen4/
│   └── ...
├── 📂 human/               
│   ├── 📂 gpt-4o/
│   ├── 📂 imagen4/
│   └── ...
├── 📂 object/                
│   └── ...
├── 📂 text/                  
│   └── ...
├── 📂 reasoning/             
│   └── ...
└── 📂 multilingualism/        # For OneIG-Bench-ZH
    └── ...
```

## Evaluation

### Parameters Configuration for Evaluation

To ensure that the generated images are correctly loaded for evaluation, you can modify the following parameters in the evaluation scripts:

1. **`mode`** : Select whether **EN/ZH** to evaluate on **OneIG-Bench** or **OneIG-Bench-ZH**.  

2. **`image_dir`** : The directory where the images generated by your model are stored.  

3. **`model_names`** : The names or identifiers of the models you want to evaluate.  

4. **`image_grid`** : This corresponds to the number of images generated by the model per prompt, where a value of **1** means 1 image, **2** means 4 images, and so on.

5. **`class_items`** : The prompt categories or image sets you want to evaluate.  

## 📈 Results

We define the sets of images generated based on the OneIG-Bench prompt categories: *General Object* (**O**), *Portrait* (**P**), *Anime and Stylization* (**A**) for prompts without stylization, (**S**) for prompts with stylization, *Text Rendering* (**T**), *Knowledge and Reasoning* (**KR**), and *Multilingualism* (**L**). 

The correspondence between the evaluation metrics and the evaluated image sets in `OneIG-Bench` and `OneIG-Bench-ZH` is presented in the table below.

- **📊 Metrics and Image Sets Correspondence**

|                    |                  Alignment                  |    Text    | Reasoning |   Style   |                  Diversity                 |
|--------------------|:-------------------------------------------:|:----------:|:---------:|:---------:|:------------------------------------------:|
| **OneIG-Bench**         | **O**, **P**, **A**, **S**                   | **T**      | **KR**    | **S**     | **O**, **P**, **A**, **S**, **T**, **KR**   |
| **OneIG-Bench-ZH**     | **O**<sub>zh</sub>, **P**<sub>zh</sub>, **A**<sub>zh</sub>, **S**<sub>zh</sub>, **L**<sub>zh</sub> | **T**<sub>zh</sub> | **KR**<sub>zh</sub> | **S**<sub>zh</sub> | **O**<sub>zh</sub>, **P**<sub>zh</sub>, **A**<sub>zh</sub>, **S**<sub>zh</sub>, **L**<sub>zh</sub>, **T**<sub>zh</sub>, **KR**<sub>zh</sub> |

- **Method Comparision on OneIG-Bench:**

<p align="center">
    <img src="./assets/result.png" width="96%">
</p>

- **Method Comparision on OneIG-Bench-ZH:**

<p align="center">
    <img src="./assets/result_ZH.png" width="96%">
</p>


- **Benchmark Comparison:**

**OneIG-Bench** (also referred to as **OneIG-Bench-EN**) denotes the English benchmark set.
<p align="center">
    <img src="./assets/benchmark_comparison.png" width="86%">
</p>


## Citation
If you find our work helpful for your research, please consider citing our work.

```bibtex
@article{chang2025oneig,
  title={OneIG-Bench: Omni-dimensional Nuanced Evaluation for Image Generation}, 
  author={Jingjing Chang and Yixiao Fang and Peng Xing and Shuhan Wu and Wei Cheng and Rui Wang and Xianfang Zeng and Gang Yu and Hai-Bao Chen},
  journal={arXiv preprint arxiv:2506.07977},
  year={2025}
}
```

## Acknowledgement
We would like to express our sincere thanks to the contributors of [Qwen](https://github.com/QwenLM/Qwen2.5-VL),  [CLIP](https://github.com/openai/CLIP), [CSD_Score](https://github.com/haofanwang/CSD_Score), [DreamSim](https://github.com/ssundaram21/dreamsim), and [HuggingFace](https://huggingface.co) teams, for their open research and exploration.


