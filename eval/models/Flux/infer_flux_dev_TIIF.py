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
# 定义保存路径和模型名称

# // 添加画廊创建函数

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
def load_jsonl(file_path, encoding='utf-8'):
    """
    加载JSON Lines格式的文件
    
    参数:
        file_path: str - JSON Lines文件的路径
        encoding: str - 文件编码，默认为'utf-8'
    
    返回:
        list - 包含文件中所有JSON对象的列表
    
    异常:
        FileNotFoundError - 当文件不存在时
        json.JSONDecodeError - 当文件中包含无效的JSON格式时
        UnicodeDecodeError - 当文件编码不正确时
    """
    json_objects = []
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                try:
                    json_obj = json.loads(line)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"在第{line_num}行解析JSON时出错: {e.msg}",
                        e.doc, e.pos
                    )
        
        return json_objects
        
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except UnicodeDecodeError:
        raise UnicodeDecodeError(encoding, b"", 0, 0, f"无法使用{encoding}编码解码文件: {file_path}")
# 设置随机种子（保证各进程采样一致）
random.seed(42)
torch.manual_seed(42)

# 初始化分布式环境
def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 修改后的PromptDatasetLeX类
class PromptDatasetTIIF(Dataset):
    def __init__(self, json_path, output_dir):
        self.output_dir = output_dir
        self.data = []
        
        # 读取JSON文件
        data_list = load_jsonl(json_path)
        
        # 处理数据
        for idx, item in enumerate(data_list):
            prompt = item.get("long_description")  # 使用long_description字段
            
            # 为每个item生成一个id（使用索引）
            prompt_id = str(idx)
            
            # 跳过缺少关键字段的条目
            if prompt is None:
                continue
            
            # 检查当前prompt的4张图像是否已全部生成（避免重复）
            all_exist = True
            for repeat_id in range(1, 5):  # repeat_id=1~4
                img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                if not os.path.exists(img_path):
                    all_exist = False
                    break
            if not all_exist:
                self.data.append({"prompt": prompt, "id": prompt_id})  # "id"对应collate_fn中的batch_ids
        
        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "id": item["id"],  # 返回prompt_id（用于文件名）
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
    # 修复键名：'annotation' → 'prompt'，匹配 __getitem__ 返回的字典结构
    prompts = [item['prompt'] for item in batch]  
    ids = [item['id'] for item in batch]
    return prompts, ids
def main():
    # 初始化分布式环境
    local_rank = setup_distributed()
    world_size = dist.get_world_size()
    print(f"Total number of processes: {world_size}")
    
    # 修改：解析命令行参数（指定 JSONL 路径和输出目录）
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Flux", 
        help="模型名称（用于输出路径子目录）"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="black-forest-labs/FLUX.1-dev", 
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        default=None, 
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--gallery_output_dir", 
        type=str, 
        default="TIIF-Bench/eval_results",
        help="图像保存根目录"
    )
    args = parser.parse_args()

    DATA = {
        'text': 'TIIF-Bench/text_prompts.jsonl'
    }   
    pipe = load_model(local_rank, args.model_path, args.lora_path)
    for key, value in DATA.items():
        output_dir = os.path.join(args.gallery_output_dir, f"{args.model_name}", key)
        if local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        dist.barrier()  # 等待所有进程完成目录创建

        # 加载模型和数据集
        
        dataset = PromptDatasetTIIF(value, output_dir)
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=16,  # 可根据GPU内存调整
            sampler=sampler,
            collate_fn=custom_collate_fn
        )

        # 推理循环（每个 prompt 生成 4 次，每次保存为独立 PNG）
        for batch_infer_idx, (batch_prompts, batch_ids) in tqdm(
            enumerate(dataloader), 
            desc=f'Processing batches {key}'
        ):
            # 循环 4 次生成（repeat_id=1~4）
            for repeat_id in range(1, 5):
                with torch.no_grad():
                    batch_images = pipe(
                        batch_prompts,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50,
                        max_sequence_length=512,
                    ).images

                # 保存当前 repeat_id 的所有图像（{prompt_id}_{repeat_id}.png）
                for img, prompt_id in zip(batch_images, batch_ids):
                    img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                    img.save(img_path, "PNG")  # 保存为 PNG 格式

            print(f"Batch {batch_infer_idx} 处理完成（{key}）")

        print(f"{key} 数据集推理完成，图像保存至: {output_dir}")


if __name__ == "__main__":
    main()