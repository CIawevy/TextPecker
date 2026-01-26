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
from PIL import Image
# 加载PEFT格式LoRA权重到transformer
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
REPEAT=5
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
            for repeat_id in range(1, REPEAT):  # repeat_id=1~4
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

def load_model(local_rank, model_path, lora_path=None):
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
        default="SD3.5", 
        help="模型名称（用于输出路径子目录）"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="stabilityai/stable-diffusion-3.5-medium", 
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
    # 加载模型和数据集
    pipe = load_model(local_rank, args.model_path, args.lora_path)

    # 对三个子集循环生成
    for key, value in DATA.items():
        # 创建输出目录（模型名称+语言，如 SD3.5_EN）
        output_dir = os.path.join(args.gallery_output_dir, f"{args.model_name}", key)
        if local_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        dist.barrier()  # 等待所有进程完成目录创建

        dataset = PromptDatasetTIIF(value, output_dir)
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=16,  # 可根据GPU内存调整
            sampler=sampler,
            collate_fn=custom_collate_fn
        )

        # 推理循环（每个 prompt 生成 1 次，每次保存为独立 PNG）
        for batch_infer_idx, (batch_prompts, batch_ids) in tqdm(
            enumerate(dataloader), 
            desc=f'Processing batches {key}'  # 替换 args.mode 为 key
        ):
            # 循环 4 次生成（repeat_id=1~4）
            for repeat_id in range(1, REPEAT):
                with torch.no_grad():
                    batch_images = pipe(
                        batch_prompts,
                        height=1024,
                        width=1024,
                        guidance_scale=3.5,
                        num_inference_steps=50
                    ).images

                # 保存当前 repeat_id 的所有图像（{prompt_id}_{repeat_id}.png）
                for img, prompt_id in zip(batch_images, batch_ids):
                    img_path = os.path.join(output_dir, f"{prompt_id}_{repeat_id}.png")
                    img.save(img_path, "PNG")  # 保存为 PNG 格式

            print(f"Batch {batch_infer_idx} 处理完成（{key}）")  # 替换 args.mode 为 key

        print(f"{key} 数据集推理完成，图像保存至: {output_dir}")  # 替换 args.mode 为 key

if __name__ == "__main__":
    main()