import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["DIFFUSERS_DISABLE_NATIVE_ATTENTION"] = "1"
import json
from diffusers import DiffusionPipeline
import torch
from datasets import load_dataset
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import random
from tqdm import tqdm 
import argparse
from safetensors.torch import load_file
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
# 定义保存路径和模型名称
# // 添加画廊创建函数
# def create_image_gallery(images, rows=2, cols=2):
#     assert len(images) >= rows * cols, "Not enough images provided!"
#     img_width, img_height = images[0].size
#     gallery_width = cols * img_width
#     gallery_height = rows * img_height
#     gallery_image = Image.new("RGB", (gallery_width, gallery_height))
#     for row in range(rows):
#         for col in range(cols):
#             img = images[row * cols + col]
#             x_offset = col * img_width
#             y_offset = row * img_height
#             gallery_image.paste(img, (x_offset, y_offset))
#     return gallery_image
def create_image_gallery(images, rows=2, cols=2, target_size=(1024, 1024)):
    assert len(images) >= rows * cols, "Not enough images provided!"
    
    # 调整所有图像到目标尺寸
    resized_images = [img.resize(target_size, Image.LANCZOS) for img in images]
    
    img_width, img_height = target_size
    gallery_width = cols * img_width
    gallery_height = rows * img_height
    gallery_image = Image.new("RGB", (gallery_width, gallery_height))
    
    for row in range(rows):
        for col in range(cols):
            img = resized_images[row * cols + col]
            x_offset = col * img_width
            y_offset = row * img_height
            gallery_image.paste(img, (x_offset, y_offset))
    
    return gallery_image


def load_model(local_rank, model_path, ckpt_path=None,use_lora=True):
    # 1. 配置设备和数据类型（对齐train_qwenimage.py）
    torch_dtype = torch.bfloat16
    device = f"cuda:{local_rank}"
    
    # 2. 加载Qwen-Image基础模型
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    ).to(device)

    pipe.safety_checker = None
    pipe.set_progress_bar_config(
        position=1,
        disable=local_rank != 0,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    if ckpt_path is not None and os.path.exists(ckpt_path) and use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
       #flow_grpo qwenimage lora loading code
        pipe.transformer = get_peft_model(pipe.transformer, transformer_lora_config)
   
        print(f'Loading SFT checkpoint from: {ckpt_path}')
        model_state_dict = load_file(ckpt_path, device="cpu")
        
        # 关键修复1：处理FSDP保存的键前缀（train_qwenimage.py第378行逻辑）
        # new_state_dict = {}
        new_state_dict = model_state_dict
        # for k, v in model_state_dict.items():
        #     new_key = k
        #     # if k.startswith('_orig_mod.'):
        #     #     new_key = k[len('_orig_mod.'):]
        #     if k.startswith('base_model.model.'):
        #         new_key = k[len('base_model.model.'):]            
        #     new_state_dict[new_key] = v
        
        # 关键修复2：加载到transformer而非unet（匹配训练时的FSDP封装对象）
        model_keys = set(pipe.transformer.state_dict().keys())
        checkpoint_keys = set(new_state_dict.keys())
        
        # 计算差异
        model_extra_keys = model_keys - checkpoint_keys
        checkpoint_extra_keys = checkpoint_keys - model_keys
        
        # 修复：将集合转换为列表后再切片
        model_extra_list = sorted(model_extra_keys)
        checkpoint_extra_list = sorted(checkpoint_extra_keys)
        
        # 打印差异结果
        print("\n=== 权重键差异分析 ===")
        print(f"模型有而checkpoint无的键: {len(model_extra_keys)}")
        # 只显示前5个差异键
        for key in model_extra_list[:5]:
            print(f"  + {key}")
        if len(model_extra_list) > 5:
            print(f"  ... 还有 {len(model_extra_list)-5} 个键未显示")
        
        print(f"\ncheckpoint有而模型无的键: {len(checkpoint_extra_keys)}")
        for key in checkpoint_extra_list[:5]:
            print(f"  - {key}")
        if len(checkpoint_extra_list) > 5:
            print(f"  ... 还有 {len(checkpoint_extra_list)-5} 个键未显示")
        
        # 加载处理后的状态字典到transformer
        msg = pipe.transformer.load_state_dict(new_state_dict, strict=False)
        print(f'Loaded checkpoint with message: {msg}')
    
    return pipe

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
     # 新增：解析命令行参数（控制CH/EN模式）
    parser = argparse.ArgumentParser()
    positive_magic = {
        "EN": "Ultra HD, 4K, cinematic composition.",
        "ZH": "超清，4K，电影级构图"
    }
    negative_prompt = " "
    aspect_ratios = {
        "1:1": (1328, 1328),  # Qwen-Image 推荐分辨率（来自 trail.py）
        "16:9": (1664, 928),
        "9:16": (928, 1664),
    }
    width, height = aspect_ratios["1:1"]
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwenimage", 
        help="Name of the model used for output directory"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen-Image", 
        help="Name of the model used for output directory"
    )
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default=None, 
        help="Path to PEFT LoRA weights directory"
    )
    parser.add_argument("--gallery_output_dir", type=str, default="TextCrafter/CVTG-2K/eval_results", help="输出根目录")
    args = parser.parse_args()

    use_lora=True
    if args.ckpt_path.strip() == "" or not os.path.exists(args.ckpt_path):
        use_lora = False
    pipe = load_model(local_rank, args.model_path,args.ckpt_path,use_lora)
    batch_size = 1
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
                # enhanced_prompts = [p + positive_magic['EN'] for p in batch_prompts]
                enhanced_prompts = batch_prompts
                # random_seed = random.randint(0, 2**32 - 1)
                # 生成图像
                with torch.no_grad():
                    batch_images = pipe(
                        prompt=enhanced_prompts,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=50,  # 推理步数（与 trail.py 一致）
                        true_cfg_scale=4.0,       # Qwen-Image 专用 CFG 参数
                        # generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(42)  # 固定种子
                        generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(42)
                    ).images
               

                # 保存图像（使用索引作为文件名，PNG格式）
                for img, idx in zip(batch_images, batch_indices):
                    img_path = os.path.join(output_dir, f"{idx}.png")
                    img.save(img_path, "PNG")  # 直接保存为PNG，不生成画廊

            print(f"Finish {benchmark} area {area} (rank {local_rank})!")



if __name__ == "__main__":
    main()