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
REPEAT = 2
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

# 修改后的PromptDatasetLeX类
class PromptDatasetAtlas(Dataset):
    def __init__(self, json_path, output_dir):
        self.output_dir = output_dir
        self.data = []
        
        # 读取JSON文件
        data_list = load_json(json_path)
        if data_list is None:
            return
        
        # 处理数据
        for item in data_list:
            prompt = item.get("annotation")  # 使用enhanced_caption字段
            prompt_id = item.get("id")  # 使用id字段
            
            # 跳过缺少关键字段的条目
            if prompt is None or prompt_id is None:
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
        default="TextAtlasEval/eval_results", 
        help="图像保存根目录"
    )
    args = parser.parse_args()

    DATA = {
        'cleantextsynth': 'TextAtlasEval/cleantextsynth.json',
        'styledtextsynth': 'TextAtlasEval/styledtextsynth.json',
        'textsceneshq': 'TextAtlasEval/textsceneshq.json',
        'textvisionblend': 'TextAtlasEval/textvisionblend.json'
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

        dataset = PromptDatasetAtlas(value, output_dir)
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
            desc=f'Processing batches {key}'  # 替换 args.mode 为 key
        ):
            # 循环 1次生成，atlas4000张重复以后太大了推理测试成本太高
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