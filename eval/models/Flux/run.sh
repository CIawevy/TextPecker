#!/bin/bash
source /mnt/bn/ocr-doc-nas/zhuhanshen/home/anaconda3/etc/profile.d/conda.sh # replace with your own
conda activate TextPeckerEval
cd /mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/eval #replace with your own
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
MODE=('EN') #EN or ZH
MODEL_PATH='/mnt/bn/ocr-doc-nas/zhuhanshen/models/FLUX.1-dev'
MODEL_NAME=(
'Flux'
'Flux-MRTPA'
'Flux-MROPA'
)
LORA_PATH=(
None
'/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/ckpts/FLUX/MRTPA_5212/lora' #no lang prefix
'/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/ckpts/FLUX/MROPA_712/lora'
)

if [ "${#MODEL_NAME[@]}" -ne "${#LORA_PATH[@]}" ]; then
    echo "Error: Number of model names doesn't match number of LoRA paths"
    exit 1
fi

for lang in "${MODE[@]}"  # 修复：使用双引号避免字段分割

do
    # Loop through parallel arrays using index loop
    for ((i=0; i<"${#MODEL_NAME[@]}"; i++)); do
        model_name="${MODEL_NAME[$i]}"
        lora_path="${LORA_PATH[$i]}"
        
        # 核心条件判断：当lora_path是None或空字符串时，lora_arg为空；否则为--lora_path "真实路径"
        [ "$lora_path" = "None" -o -z "$lora_path" ] && lora_arg="" || lora_arg="--lora_path $lora_path"
        
        # Generate a new port for each model combination
        FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
        
        echo "=========================================================="
        echo "Running inference for model: $model_name with LoRA: $lora_path"
        echo "Using port: $FREE_PORT"
        echo "Current language: $lang"  # 修复：添加当前语言显示
        echo "=========================================================="
        
        #OneIG infer
        # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_oneig.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #CVTG2K infer
        # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_CVTG2K.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #LongText infer
        # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_longtext.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #LeX infer
        # # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_LeX.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #TIIF infer
        # # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_TIIF.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #Atlas infer
        # # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_TextAtlasEval.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        # #GenTextEval infer
        # torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_gentexteval.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg
    done
done