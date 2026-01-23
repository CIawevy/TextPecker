#!/bin/bash
source your_conda_env/bin/activate  # replace with your own
conda activate TextPecker
cd TextPecker/eval #replace with your own
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
MODE=('EN') #EN or ZH
MODEL_PATH='YOUR MODEL PATH' #Qwen/Qwen-Image
MODEL_NAME=(
'MODEL NAME'   #no lang sufix
'MODEL NAME 1' #no lang sufix
)
LORA_PATH=(
"" #default setting 
'YOUR LORA PATH' 
)

if [ "${#MODEL_NAME[@]}" -ne "${#LORA_PATH[@]}" ]; then
    echo "Error: Number of model names doesn't match number of LoRA paths"
    exit 1
fi

for lang in "${MODE[@]}"  
do
    # Loop through parallel arrays using index loop
    for ((i=0; i<"${#MODEL_NAME[@]}"; i++)); do
        model_name="${MODEL_NAME[$i]}"
        lora_path="${LORA_PATH[$i]}"
        
        # 核心条件判断：当lora_path是None或空字符串时，lora_arg为空；否则为--lora_path "真实路径"
        [ "$lora_path" = "None" -o -z "$lora_path" ] && lora_arg="" || lora_arg="--lora_path '$lora_path'"
        
        # Generate a new port for each model combination
        FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
        
        echo "=========================================================="
        echo "Running inference for model: $model_name with LoRA: $lora_path"
        echo "Using port: $FREE_PORT"
        echo "Current language: $lang"  
        echo "=========================================================="
        
        #OneIG infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_oneig.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #CVTG2K infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_CVTG2K.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #LongText infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_longtext.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #GenTextEval infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_gentexteval.py --mode $lang --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #LeX infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_LeX.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #Atlas infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_TextAtlasEval.py --model_name $model_name --model_path $MODEL_PATH $lora_arg

        #TIIF infer
        torchrun --nproc_per_node=8 --master-port $FREE_PORT models/Flux/infer_flux_dev_TIIF.py --model_name $model_name --model_path $MODEL_PATH $lora_arg
    done
done