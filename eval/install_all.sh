# please use python=3.10/3.11, cuda12.*

# sh eval/install_all.sh
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 -U #cuda 12.4 as example
pip install "transformers==4.54.0" peft -U 
# flash-attn: https://github.com/Dao-AILab/flash-attention/releases
# diffusers
pip install git+https://github.com/huggingface/diffusers.git
pip install flash_attn==2.6.1 --no-build-isolation #cuda 12.4 as example