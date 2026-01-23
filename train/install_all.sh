# please use python=3.10/3.11, cuda12.*
#qwen3vl: transformers>=4.57, qwen_vl_utils>=0.0.14, decord torch<2.9 
#internvl 3.5: ransformers>=4.37.2, timm 

# sh requirements/install_all.sh
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 -U #cuda 12.4 as example
# pip install "sglang<0.5.6" -U
pip install "vllm>=0.5.1,<0.11.1" -U
# pip install "lmdeploy>=0.5,<0.10.2" -U
pip install "transformers>=4.57,<4.58" "trl<0.25" peft -U #support qwen3vl
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]
pip install timm "deepspeed<0.18" -U
pip install qwen_vl_utils qwen_omni_utils keye_vl_utils -U
pip install decord librosa icecream soundfile -U
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
pip install flash_attn==2.8.3 --no-build-isolation #cuda 12.4 as example
pip install megfile #eval
pip install asyncio #eval

