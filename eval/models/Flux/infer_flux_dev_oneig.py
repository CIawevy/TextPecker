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
import megfile
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
class PromptDatasetOneIG(Dataset):
    def __init__(self, data, output_dir=None):
        self.output_dir = output_dir
        self.data = []

        # 处理不同数据格式
        if isinstance(data, dict):
            data = list(data.values())
        elif not isinstance(data, list):
            raise ValueError("Unsupported data type")

        exitsnums = 0
        # 筛选Text_Rendering类别样本
        for item in data:
            if item.get('category') == 'Text_Rendering':
                self.data.append(item)

        # 排除已存在图像的样本
        if self.output_dir:
            filtered_data = []
            for item in self.data:
                image_path = os.path.join(self.output_dir, f"{item['id']}.webp")
                if not os.path.exists(image_path):
                    filtered_data.append(item)
                else:
                    exitsnums += 1
                    print(f'image_path:{image_path} exists thus skip')
             
            self.data = filtered_data

        print(f"Loaded {len(self.data)} Text_Rendering samples,exisiting {exitsnums} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item.get('prompt_en') or item.get('prompt_cn'),
            'id': item['id'],
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
        print(f'############################start load lora_path:{lora_path}#########################')
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
     # 新增：解析命令行参数（控制CH/EN模式）
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["ZH", "EN"], 
        default="EN", 
        help="Dataset mode: ZH (Chinese) or EN (English)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Flux", 
        help="Name of the model used for output directory"
    )
    parser.add_argument(
        "--lora_path", 
        type=str, 
        default=None, 
        help="Path to PEFT LoRA weights directory"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="black-forest-labs/FLUX.1-dev", 
        help="Name of the model used for output directory"
    )
    parser.add_argument(
        "--gallery_output_dir", 
        type=str, 
        default="OneIG-Benchmark/images/text/", 
        help="Name of the model used for output directory"
    )
    args = parser.parse_args()

    # 根据mode选择数据集路径
    if args.mode == "EN":
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json"
    else:  # ZH模式
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench-ZH.json"
    batch_size = 16
    # 加载模型

    output_dir = os.path.join(args.gallery_output_dir, args.model_name+f'_{args.mode}')
    if local_rank == 0:  # 仅主进程创建文件夹
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()  # 等待主进程完成文件夹创建
    if args.lora_path is not None and args.lora_path.strip() != "":
        print(f'load lora_path:{args.lora_path}')
    else:
        print(f'no lora_path')
    pipe = load_model(local_rank, args.model_path,args.lora_path)

    ds = load_json(load_json_file)
    dataset = PromptDatasetOneIG(ds, output_dir=output_dir)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,collate_fn=custom_collate_fn)
    for batch_infer_idx,(batch_prompts, batch_ids) in tqdm(enumerate(dataloader),desc=f'Processing batches OneIG-{args.mode}'):
        prompt_images = [[] for _ in range(len(batch_prompts))]
        # 为每个prompt生成4张图像
        # 循环4次推理：每次处理16个prompt，收集图像到对应子列表
        for i in range(4):
            print(f'gen prompt idx :{i}')
            with torch.no_grad():
                # 单次推理：batch_size=16，返回16张图像（与batch_prompts顺序对应）
                batch_images = pipe(
                    batch_prompts,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=50,
                    max_sequence_length=512,
                ).images
            # 将当前批次图像分配到对应prompt的子列表（第i个图像→第i个prompt的子列表）
            for i, img in enumerate(batch_images):
                # print(f'len:{len(batch_images)}')
                prompt_images[i].append(img)  # 每个子列表逐步收集4张图像
        print(f'len(prompt_images):{len(prompt_images)}')
        # 新增：在第一个批次保存第一张prompt的4张生成图用于差异性验证
        # if batch_infer_idx == 0:
        #     # 选择第一张prompt的4张生成图像
        #     idx = random.randint(0, len(batch_prompts)-1)
        #     print(f'prompt:{batch_prompts[idx]}')
        #     sample_prompt_images = prompt_images[idx]
        #     # 保存路径
        #     vis_dir = "eval/OneIG-Benchmark/images/text/"
        #     os.makedirs(vis_dir, exist_ok=True)
        #     # 分别保存4张图像
        #     for i, img in enumerate(sample_prompt_images, 1):
        #         img_path = os.path.join(vis_dir, f"temp{i}.png")
        #         img.save(img_path, "PNG")
        #     print(f"已保存验证图像到: {vis_dir}")

        # 循环处理16个prompt，创建画廊并保存
        for i in range(len(batch_prompts)):
            # 获取当前prompt的4张图像（已通过4次循环收集完成）
            current_images = prompt_images[i]
            # 确保有足够的图像创建画廊
            if len(current_images) < 4:
                print(f"警告: prompt {i} 只生成了 {len(current_images)} 张图像，需要4张。使用现有图像创建画廊。")
                # 如果图像不足，重复使用现有图像填充
                while len(current_images) < 4:
                    current_images.append(current_images[-1])
            else:
                print(f'已有四张图，开始保存')
            # 创建2x2画廊
            gallery = create_image_gallery(current_images, rows=2, cols=2)
            # 保存：使用batch_ids[i]作为唯一ID（对应原数据集中的id字段）
            file_path = megfile.smart_path_join(output_dir, f"{batch_ids[i]}.webp")
            with megfile.smart_open(file_path, "wb") as f:
                gallery.save(f, "WebP")

    print(f'finish OneIG-{args.mode} !')

if __name__ == "__main__":
    main()