## How to Use Our Developed Data Engine

Our data engine is **built on top of SynthTiger**.  
On this basis, we add:

1. Rendering **bilingual, multi-line text** directly into custom images with various transformations and effects.
2. **Structural editing** on synthesized content to create structual-anomaly rendering data for Chinese.

All paths below are relative to this directory: `TextPecker/data`.

---

### 1. Prepare Data

Use the folders under `./synthtiger/resources` and replace the toy data with your own:

- Fonts: `./synthtiger/resources/fonts`
- Corpora: `./synthtiger/resources/corpus`
- Background images: `./synthtiger/resources/bg-images` 

Keep the file formats and structure consistent with the original SynthTiger/Ours settings.  
We provide toy data in the code as examples.

---

### 2. Environment Setup

```bash
cd ./synthtiger
conda create -n tiger python=3.11 -y
conda activate tiger
pip install -e .
```
This will install our modified SynthTiger together with all pinned dependencies defined in `./synthtiger/setup.py`

---

### 3. Run the Pipeline

Modify and Run the following scripts to try our developed data engine.

#### Step 1: Synthesize Data

```bash
bash ./synthtiger/examples/run.sh
```

#### Step 2: Merge JSONL Files

```bash
bash ./synthtiger/merge_jsonl.sh
bash ./synthtiger/merge_jsonl_zaozi.sh
```

#### Step 3: Convert to QA Format

```bash
python ./synthtiger/make_qa_syndata.py
python ./synthtiger/make_qa_syndata_zaozi.py
```

You can customize these scripts and corresponding configs to match your own data and paths.