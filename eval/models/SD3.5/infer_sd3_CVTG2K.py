
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import json
import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from datasets import load_dataset
import random
from tqdm import tqdm
import argparse
import megfile
from PIL import Image
# 加载PEFT格式LoRA权重到transformer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel


def load_json(file_path):
    """
    加载指定路径的 JSON 文件并将其内容解析为 Python 对象。

    :param file_path: 要加载的 JSON 文件的路径
    :return: 解析后的 Python 对象，如果出现错误则返回 None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误：指定的文件 {file_path} 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误：无法解析 {file_path} 中的 JSON 数据。")
        return None
    except Exception as e:
        print(f"发生未知错误：{e}")
        return None

# 设置随机种子（保证各进程采样一致）
random.seed(42)
torch.manual_seed(42)

# 初始化分布式环境
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
# 新增：CVTG2K 数据集类
class PromptDatasetCVTG2K(Dataset):
    def __init__(self, benchmark, area, output_dir):
        self.output_dir = output_dir
        self.data = []
        
        # 加载 CVTG JSON 数据（按 benchmark 和 area 区分）
        json_path = f"TextCrafter/CVTG-2K/{benchmark}/{area}.json"
        json_data = load_json(json_path)
        if not json_data:
            raise ValueError(f"Failed to load data from {json_path}")
        
        # 提取 data_list 并按索引顺序排序
        data_list = json_data.get("data_list", [])
        data_list.sort(key=lambda x: x["index"])  # 确保索引顺序提取
        
        # 筛选未生成图像的样本
        for item in data_list:
            index = item.get("index")
            prompt = item.get("prompt")
            if index is not None and prompt:
                # 检查图像是否已存在
                image_path = os.path.join(output_dir, f"{index}.png")
                if not os.path.exists(image_path):
                    self.data.append({"prompt": prompt, "index": index})
        
        print(f"Loaded {len(self.data)} CVTG2K samples (benchmark: {benchmark}, area: {area})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "index": item["index"],  # 顺序提取索引
        }
# # 加载SD3模型到当前GPU
# def load_model(local_rank, model_path,lora_path=None):
#     pipe = StableDiffusion3Pipeline.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16
#     ).to(f"cuda:{local_rank}")
#     if lora_path is not None:
#         pipe.transformer = PeftModel.from_pretrained(
#             pipe.transformer,
#             lora_path,
#             device_map=f"cuda:{local_rank}"
#         )
#      # 可选：启用推理模式
#     pipe.transformer.eval()
#     return pipe
def load_model(local_rank, model_path,lora_path=None):
    # inference_dtype = torch.bfloat16
    device = f"cuda:{local_rank}"
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)
    # load scheduler, tokenizer and models.

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not local_rank==0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.


    # Move vae and text_encoder to device and cast to inference_dtype

    pipeline.transformer.to(device)
    if lora_path is not None and lora_path.strip() != "" and os.path.exists(lora_path):
        pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, lora_path)
      
     
    pipeline.transformer.eval()
    return pipeline
def custom_collate_fn(batch):
    # 修改：从 'index' 字段提取索引（原 'id' 改为 'index'）
    prompts = [item["prompt"] for item in batch]  
    indices = [item["index"] for item in batch]  # 按顺序提取索引
    return prompts, indices
def main():
    # 初始化分布式环境
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    print(f"Total number of processes: {world_size}")
    
    # 新增：解析命令行参数（补充模型名称和输出路径）
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="SD3.5", help="模型名称，用于输出路径")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="基础模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA权重路径")
    parser.add_argument("--gallery_output_dir", type=str, default="TextCrafter/CVTG-2K/eval_results", help="输出根目录")
    args = parser.parse_args()

    # 加载模型
    pipe = load_model(local_rank, args.model_path, args.lora_path)
    batch_size = 16  # 根据GPU内存调整

    # 新增：循环处理 CVTG 和 CVTG-Style（多卡推理核心逻辑）
    for area in range(2, 6):  # 遍历 area=2,3,4,5
        for benchmark in ("CVTG", "CVTG-Style"):  # 遍历两类数据集
            # 构建输出路径：模型名称/数据集类型/区域 (如 SD3.5/CVTG/2)
            output_dir = os.path.join(args.gallery_output_dir, args.model_name, benchmark, str(area))
            if local_rank == 0:  # 主进程创建目录
                os.makedirs(output_dir, exist_ok=True)
            dist.barrier()  # 等待所有进程完成目录创建

            # 加载 CVTG2K 数据集
            dataset = PromptDatasetCVTG2K(benchmark, area, output_dir)
            sampler = DistributedSampler(dataset, shuffle=False)  # 多卡分布式采样
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                collate_fn=custom_collate_fn
            )

            # 推理与保存（按批次处理）
            for batch_infer_idx, (batch_prompts, batch_indices) in tqdm(
                enumerate(dataloader), 
                desc=f"Processing {benchmark} area {area} (rank {local_rank})"
            ):
                # 单批次推理（不再生成4张图像，每张prompt生成1张）
                with torch.no_grad():
                    batch_images = pipe(
                        batch_prompts,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50
                    ).images

                # 保存图像（使用索引作为文件名，PNG格式）
                for img, idx in zip(batch_images, batch_indices):
                    img_path = os.path.join(output_dir, f"{idx}.png")
                    img.save(img_path, "PNG")  # 直接保存为PNG，不生成画廊

            print(f"Finish {benchmark} area {area} (rank {local_rank})!")

if __name__ == "__main__":
    main()

# ... 移除原有的main函数定义（避免冲突）...

