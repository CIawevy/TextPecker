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
import megfile
from safetensors.torch import load_file
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
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
    
    if ckpt_path is not None and use_lora:
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
        "--mode", 
        type=str, 
        choices=["ZH", "EN"], 
        default="EN", 
        help="Dataset mode: CH (Chinese) or EN (English)"
    )
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
    parser.add_argument(
        "--gallery_output_dir", 
        type=str, 
        default="OneIG-Benchmark/images/text", 
        help="Name of the model used for output directory"
    )
    args = parser.parse_args()

    # 根据mode选择数据集路径
    if args.mode == "EN":
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json"
    else:  # CH模式
        load_json_file = "OneIG-Benchmark/OneIG-Bench/OneIG-Bench-ZH.json"
    batch_size = 1
    # 创建保存图像的文件夹
    output_dir = os.path.join(args.gallery_output_dir, args.model_name+f'_{args.mode}')
    if local_rank == 0:  # 仅主进程创建文件夹
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()  # 等待主进程完成文件夹创建
    # 加载模型
    use_lora = True
    ckpt_path = args.ckpt_path
    if (not ckpt_path) or (ckpt_path.strip() == "") or (not os.path.exists(ckpt_path)):
        use_lora = False
        ckpt_path = None
    pipe = load_model(local_rank, args.model_path, ckpt_path, use_lora)

    ds = load_json(load_json_file)
    dataset = PromptDatasetOneIG(ds, output_dir=output_dir)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,collate_fn=custom_collate_fn)
    for batch_infer_idx,(batch_prompts, batch_ids) in tqdm(enumerate(dataloader),desc=f'Processing batches OneIG-{args.mode}'):
        # enhanced_prompts = [p + positive_magic[args.mode] for p in batch_prompts]
        enhanced_prompts = batch_prompts
        prompt_images = [[] for _ in range(len(batch_prompts))]
        # 为每个prompt生成4张图像
        # 循环4次推理：每次处理16个prompt，收集图像到对应子列表
        for i in range(4):
            print(f'gen prompt idx :{i}')
            random_seed = random.randint(0, 2**32 - 1)
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
                    generator=torch.Generator(device=f"cuda:{local_rank}").manual_seed(random_seed)
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
        #     vis_dir = "eval/models/Qwenimage"
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