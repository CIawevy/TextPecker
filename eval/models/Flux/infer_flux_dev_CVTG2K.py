import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import torch
from diffusers import FluxPipeline
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import random
from tqdm import tqdm
import argparse
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel

# // 添加画廊创建函数
def create_image_gallery(images, rows=2, cols=2):
    assert len(images) >= rows * cols, "Not enough images provided!"
    img_width, img_height = images[0].size
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))
    for row in range(rows):
        for col in range(cols):
            img = images[row * cols + col]
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))
    return gallery_image
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
        json_path = f"TextCrafter/CVTG-2K/{benchmark}/{area}.json" #replace with your own
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
# 加载模型到当前 GPU
def load_model(local_rank, model_path,lora_path=None):
    pipeline = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{local_rank}")
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not local_rank==0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # 加载PEFT LoRA权重
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
    parser.add_argument("--model_name", type=str, default="Flux", help="模型名称，用于输出路径")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-dev", help="基础模型路径")
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
                        num_inference_steps=50,
                        max_sequence_length=512,
                    ).images

                # 保存图像（使用索引作为文件名，PNG格式）
                for img, idx in zip(batch_images, batch_indices):
                    img_path = os.path.join(output_dir, f"{idx}.png")
                    img.save(img_path, "PNG")  # 直接保存为PNG，不生成画廊

            print(f"Finish {benchmark} area {area} (rank {local_rank})!")

if __name__ == "__main__":
    main()


    
