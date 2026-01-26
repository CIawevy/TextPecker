import os
# os.environ['no_proxy'] = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b"
import sys 
# 将当前文件的上级目录设置为sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import json
import argparse
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import io
import numpy as np
import torch
import hashlib
from collections import defaultdict
import asyncio
from openai import AsyncOpenAI
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from parse_utils_pecker import get_score_v2
from typing import Union
import time
import threading
import os
import json
from concurrent import futures
import time
from typing import Union
# 添加tqdm库用于进度条可视化
from tqdm import tqdm
import pandas as pd
import datetime
import stat

# 定义全局元数据配置变量
META_DATA = {
    'longtext': {
        'en': 'X-Omni/textbench/text_prompts.jsonl',
        'zh': 'X-Omni/textbench/text_prompts_zh.jsonl'
    },
    # 可以根据需要轻松添加其他数据集类型
    'cvtg': {
        'en': 'TextCrafter/CVTG-2K',
    },
    'oneig': {
        'en': 'OneIG-Benchmark/scripts/text/text_content.csv',
        'zh': 'OneIG-Benchmark/scripts/text/text_content_zh.csv'
    },
    'gentexteval':{
        'en' : 'GenTextEval/GenTextEval_en.jsonl',
        'zh' : 'GenTextEval/GenTextEval_zh.jsonl'
    },
    'lex':{
        'en' : {
            'easy': 'LeX-Bench/lex_bench_easy.json',
            'medium': 'LeX-Bench/lex_bench_medium.json',
            'hard': 'LeX-Bench/lex_bench_hard.json'
        }
    } ,
    'atlas':{
        'en' : {
            'cleantextsynth': 'TextAtlasEval/cleantextsynth.json',
            'styledtextsynth': 'TextAtlasEval/styledtextsynth.json',
            'textsceneshq': 'TextAtlasEval/textsceneshq.json',
            'textvisionblend': 'TextAtlasEval/textvisionblend.json'
        }
    } ,
    'tiif':{
        'en' : {
            'text': 'TIIF-Bench/text_prompts.jsonl'
        }
    }
}
# --- 1. VLLM Server Configuration ---
PORT = 8851 #replace with your own
VLLM_HOST = "2605:340:cd60:0:90cb:803d:7dc0:99a8" #replace with your own

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


def ocr_score_mtl(device):
    #中英文的ocr score function
    from ocr import OcrScorerEN
    from ocr import OcrScorerCN

    en_scorer = OcrScorerEN()
    cn_scorer = OcrScorerCN()
    def contain_chinese(prompts):
        return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False

    def _fn(images, prompts,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 按语言类型分组索引
        # cn_indices = [i for i, prompt in enumerate(prompts) if contain_chinese(prompt)]
        # en_indices = [i for i, prompt in enumerate(prompts) if not contain_chinese(prompt)]
        cn_indices = [i for i, target in enumerate(targets) if contain_chinese(target)]
        en_indices = [i for i, target in enumerate(targets) if not contain_chinese(target)]
        
        # 初始化结果列表
        scores = [0.0] * len(prompts)
        recs = ["todo"]  * len(prompts)
        
        # 中文组推理
        if cn_indices:
            cn_images = [images[i] for i in cn_indices]
            cn_prompts = [prompts[i] for i in cn_indices]
            cn_targets = [targets[i] for i in cn_indices]
            cn_scores, cn_recs = cn_scorer(cn_images, cn_prompts,cn_targets)
            for i, idx in enumerate(cn_indices):
                scores[idx] = cn_scores[i]
                recs[idx] = cn_recs[i]
        
        # 英文组推理
        if en_indices:
            en_images =  [images[i] for i in en_indices]
            en_prompts = [prompts[i] for i in en_indices]
            en_targets = [targets[i] for i in en_indices]
            en_scores,en_recs = en_scorer(en_images, en_prompts,en_targets)
            for i, idx in enumerate(en_indices):
                scores[idx] = en_scores[i]
                recs[idx] = en_recs[i]
  
        return scores, recs

    return _fn


# 首先在文件顶部添加线程本地存储对象和客户端缓存
_thread_local = threading.local()
_process_clients = {}
_vllm_client_lock = threading.Lock()

# 修改textpecker_score_vllm函数
def textpecker_score_vllm(vllm_host=VLLM_HOST, vllm_port=PORT):
    """
    Computes an OCR-based reward score using a remote TextPecker model served with vLLM.
    """

    # --- 1. VLLM Server Configuration ---  
    # 使用传入的参数，避免与全局变量混淆
    current_vllm_host = vllm_host
    current_vllm_port = vllm_port
    print(f"Using vLLM server at {current_vllm_host}:{current_vllm_port}")
    # Format URL for IPv6 if needed
    if ":" in current_vllm_host:
        VLLM_BASE_URL = f"http://[{current_vllm_host}]:{current_vllm_port}/v1"
    else:
        VLLM_BASE_URL = f"http://{current_vllm_host}:{current_vllm_port}/v1"
    VLLM_API_KEY = "EMPTY"
    MODEL_NAME = "TextPecker"
    
    # 使用进程级客户端缓存和锁确保线程安全
    client_key = f"{VLLM_BASE_URL}_{VLLM_API_KEY}"
    if client_key not in _process_clients:
        with _vllm_client_lock:
            # 双重检查锁定模式
            if client_key not in _process_clients:
                client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
                _process_clients[client_key] = client
                    
    client = _process_clients[client_key]

    # --- 2. Helper Functions ---  
    def pil_image_to_base64(image: Union[Image.Image, str]) -> str:
        """Converts a PIL image to a base64 string for API calls."""
        if isinstance(image, str):
            image = Image.open(image)
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_image_text}"

    def get_RGI_template(target_text: str) -> str:
        """Creates the prompt for the RGI task based on the target text."""
        return f'''This is a text-generated image. Please recognize text in the image and compare it with the target text "{target_text}". 
                Marking rules: 
                1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters; 
                2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences). 
                Classify differences (output only the classification results, enclosed in quotes and separated by commas) in the following JSON format:
                {{
                "recognized_text": "All recognized text in the image (including structural error markers)",
                "duplicate_text": "Specific duplicate content",
                "missing_text": "Specific missing content",
                "typo_text": "Specific typographical errors",
                "irrelevant_text": "Specific irrelevant text",
                "correct_text": "Specific correct text"
                }}
                '''
    def get_rec_template() -> str:
        """Creates the prompt for the Recognition task"""
        question = """
        This is a text-generated image. Please recognize all visible text in the entire image.
        Marking rules: 
        1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
        2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
        Output in the following JSON format:
        {
        "recognized_text": "All text in the image (including structural error markers)"
        }
        """
        return question
    
    # --- 3. 健壮的异步任务执行函数 ---  
    def run_async_task(coro):
        """
        安全地运行异步任务，为每个线程维护独立的事件循环
        避免在多线程环境下出现 'Event loop is closed' 错误
        """
        # 检查当前线程是否已有事件循环
        if not hasattr(_thread_local, 'loop'):
            try:
                # 尝试获取当前线程的事件循环
                loop = asyncio.get_event_loop()
                # 检查事件循环是否已关闭
                if loop.is_closed():
                    # 如果已关闭，则创建新的事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                _thread_local.loop = loop
            except RuntimeError:
                # 如果没有事件循环，则创建新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop
        else:
            loop = _thread_local.loop
            # 再次检查事件循环是否已关闭
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _thread_local.loop = loop
        
        try:
            # 如果在当前线程的事件循环中运行
            if loop.is_running():
                # 使用 create_task 并等待结果
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
            else:
                # 在非运行的事件循环中直接运行
                return loop.run_until_complete(coro)
        except Exception as e:
            # 发生异常时，创建新的事件循环重试
            print(f"Error in run_async_task: {e}. Creating new event loop and retrying...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _thread_local.loop = loop
            return loop.run_until_complete(coro)

    # --- 4. Async API Call Functions with Enhanced Retry ---  
    async def evaluate_image(prompt: str, base64_image: str, max_retries=3):
        """
        带重试机制的异步图像评估函数
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                # 直接使用外部函数的client变量
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": base64_image}},
                                {"type": "text", "text": prompt},
                            ],
                        },
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    # top_p=0.6,
                    # presence_penalty=1.5,
                    extra_body={
                        # "top_k": 50, 
                        "repetition_penalty":1.2
                        # "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                return response.choices[0].message.content
            except Exception as e:
                retry_count += 1
                last_exception = e
                wait_time = min(5 * (2 ** (retry_count - 1)), 30)  # 指数退避
                print(f"Error in evaluate_image (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        # 所有重试都失败时，抛出异常而不是返回默认值
        raise RuntimeError(f"Failed to evaluate image after {max_retries} attempts") from last_exception

    async def evaluate_batch_image(prompts, images):
        """Creates and runs a batch of asynchronous evaluation tasks with proper error handling."""
        images_base64 = [pil_image_to_base64(img) for img in images]
        tasks = [evaluate_image(prompt, img_b64) for prompt, img_b64 in zip(prompts, images_base64)]
        
        # 使用 gather 但不捕获异常，确保问题能够被上层感知并处理
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, targets):
        """
        The actual reward function that will be called by the framework.
        `prompts` are the generation prompts (ignored here).
        `targets` are the ground-truth texts for OCR.
        """
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        
        # Construct the specific prompts for TextPecker using the target texts
        # vllm_prompts = [get_RGI_template(target) for target in targets]
        vllm_prompts = [get_rec_template() for target in targets]


        # 使用健壮的异步任务执行函数替代 asyncio.run
        text_outputs = run_async_task(evaluate_batch_image(vllm_prompts, images))
        
        # Parse model outputs and calculate scores using the logic from qwenvl.py
        # sem_rewards = []
        qua_rewards = []
        res = []
        gned_scores = []
        qua_amplify_factor = 1.0 # 1/5 = 20%  #1.0 for evaluation we get direct quality score without scaling or amplify
        for j,response in enumerate(text_outputs):
            try:
                # sem_score, qua_score, gned_score,rec_text, correct_text = get_score('RGI', response, targets[j])
                quality_score,gned_score,cls_results = get_score_v2( response, targets[j],qua_amplify_factor,True)
            except Exception as e:
                print(f'error:{e}')
                print(f'cur response:\n{response}')
                quality_score,gned_score,cls_results = 0, 0, {}
                
            # sem_rewards.append(sem_score)
            qua_rewards.append(quality_score)
            res.append(cls_results)
            gned_scores.append(gned_score)
        
        return  qua_rewards,  gned_scores , res

    return _fn




def detect_language(output_dir):
    """根据模型名称自动检测语言模式"""
    if '_EN' in output_dir:
        return 'en'
    elif '_ZH' in output_dir:
        return 'zh'
    else:
        # 默认返回英文
        return 'en'
    
import os
import json
import glob
from pathlib import Path

# 修改并完善 load_meta_cvtg 函数
def load_meta_cvtg(sample_dir, lang_mode, cache_dir=None):
    """加载CVTG数据集的metadata信息，根据语言模式处理"""
    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'cvtg'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取CVTG数据集的基本路径
    cvtg_base_dir = META_DATA[dataset_type][lang_mode]
    
    # 检查基本路径是否存在
    if not os.path.exists(cvtg_base_dir):
        raise FileNotFoundError(f"CVTG base directory not found: {cvtg_base_dir}")
    
    data = []
    
    # 遍历所有的benchmark类型和区域，与unified_metrics_eval.py保持一致
    for benchmark_type in ['CVTG', 'CVTG-Style']:
        for area in [2, 3, 4, 5]:
            # 构建prompt文件路径
            prompt_file = os.path.join(cvtg_base_dir, benchmark_type, f"{area}.json")
            
            # 构建图像目录路径（注意：这里使用sample_dir作为结果目录）
            image_dir = os.path.join(sample_dir, benchmark_type, str(area))
            
            # 检查prompt文件和图像目录是否存在
            if not os.path.exists(prompt_file):
                print(f"Warning: Prompt file not found: {prompt_file}, skipping...")
                continue
            
            if not os.path.exists(image_dir):
                print(f"Warning: Image directory not found: {image_dir}, skipping...")
                continue
            
            # 读取prompt信息
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                    # 创建index到prompt的映射
                    prompts = {str(item['index']): item['prompt'] for item in prompt_data.get('data_list', [])}
            except Exception as e:
                print(f"Error reading prompt file {prompt_file}: {e}, skipping...")
                continue
            
            # 获取图像文件列表
            image_files = [f for f in os.listdir(image_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 处理每个图像文件
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                img_id = Path(img_file).stem  # 获取文件名（不含扩展名）作为ID
                
                # 检查是否有对应的prompt
                if img_id not in prompts:
                    print(f"Warning: No prompt found for image {img_file}, skipping...")
                    continue
                
                prompt = prompts[img_id]
                
                # 解析prompt获取目标文本
                # 对于CVTG数据集，我们提取引号内的文本作为target
                # 可以根据实际需要调整解析逻辑
                target = parse_prompt(prompt, mode='en')
                
                #CVTG 本来是没有提取好的字段，只有Prompt，评估时直接提取引号来获取word gt，与我的parse 逻辑是一致的本质上
                # 创建data字典
                data.append({
                    'image': img_path,
                    'prompt': prompt,
                    'target': target,
                    'ori_target':target,
                    'id': f'{dataset_type}_{len(data)}'
                })
    
    return data

# 确保导入必要的模块
import os
import glob
import ast
import pandas as pd
from PIL import Image
import shutil

def split_2x2_grid(image_path, grid_size, cache_dir):
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 从图像路径生成唯一标识符，避免不同图像的结果相互覆盖
    # 使用文件名（不含扩展名）和路径的哈希值组合作为唯一标识符
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    path_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]  # 取哈希值前8位作为简短唯一标识
    unique_prefix = f"{image_basename}_{path_hash}_"
    
    # 打开原始图像
    with open(image_path, 'rb') as f:
        grid_image = Image.open(f)

        width, height = grid_image.size
        individual_width = width // grid_size[0]
        individual_height = height // grid_size[1]

        image_list = []

        # 切分图像为2x2网格
        for i in range(grid_size[1]):
            for j in range(grid_size[0]):
                box = (
                    j * individual_width,      
                    i * individual_height,     
                    (j + 1) * individual_width,  
                    (i + 1) * individual_height  
                )

                individual_image = grid_image.crop(box)
                image_list.append(individual_image)

    # 保存切分后的图像，使用唯一前缀避免覆盖
    image_path_list = []
    for i, image in enumerate(image_list):
        # 使用唯一前缀 + 索引命名文件
        image_path = os.path.join(cache_dir, f"{unique_prefix}{i}.jpg")
        image.save(image_path)
        image_path_list.append(image_path)

    return image_path_list

def load_meta_oneig(sample_dir, lang_mode,cache_dir):
    """加载oneig数据集的metadata信息，根据语言模式处理"""
    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'oneig'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取对应的prompt文件路径
    prompt_file = META_DATA[dataset_type][lang_mode]
    
    # 检查文件是否存在
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
    
    # 使用pandas加载CSV文件
    text_df = pd.read_csv(prompt_file, dtype=str)
    
    data = []
    # 确保缓存目录存在
    # os.makedirs(cache_dir, exist_ok=True)
    
    # 遍历text_df中的每一行数据
    for _, row in text_df.iterrows():
        img_id = row['id']
        if lang_mode == 'zh':
            prompt_key = 'prompt_cn'
        elif lang_mode == 'en':
            prompt_key = 'prompt_en'
        prompt = row[prompt_key]
        text_str = row['text_content']  # 读取的是字符串形式的列表，如"['a', 'b']"
        
        # 关键步骤：将字符串转换为真正的Python列表
        text_list = ast.literal_eval(text_str)
        
        # 生成目标文本（用空格连接列表元素）
        target_text = ' '.join(text_list)
        
        # 解析prompt获取目标
        target = parse_prompt(prompt, mode=lang_mode)
        if len(target)==0:
            print(f'parse failed use ori target!')
            target = target_text
        if len(target)!=0 and (len(target_text)/len(target))>=2:
            print(f'parse failed for len << ori target use ori target')
            target = target_text

        
        # 根据您提供的路径格式查找图像文件
        # 格式示例：eval/OneIG-Benchmark/images/text/Flux_EN/000.webp
        # 这里假设sample_dir已经包含了到images/text的路径
        img_pattern = os.path.join(sample_dir, f'{img_id}.webp')
        img_paths = glob.glob(img_pattern)
        
        
        
        # 对每个找到的webp文件进行处理
        for webp_file in img_paths:
            # 使用2x2网格切分图像
            split_img_list = split_2x2_grid(webp_file, (2, 2), cache_dir)
            
            # 为每张切分后的图像创建一个data字典
            for split_img_path in split_img_list:
                data.append({
                    'image': split_img_path,  # 切分后的图像路径
                    'prompt': prompt,
                    'target': target,
                    'ori_target':target_text,
                    'id' : f'{dataset_type}_{len(data)}'
                })
    
    return data



def contain_chinese(prompts):
    """检查prompt是否包含中文字符"""
    return True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False


def parse_quoted_text(prompt):
    """提取引号包裹的文本内容（支持中英文引号/书名号）"""
    quote_map = {
        '“': '”',   # 中文双引号
        '‘': '’',   # 中文单引号
        '"': '"',   # 英文双引号
        "'": "'",   # 英文单引号
        '《': '》'   # 书名号
    }
    
    results = []
    in_quote = False
    current_closing = None
    start_pos = 0
    
    for i, char in enumerate(prompt):
        if not in_quote:
            if char in quote_map:
                in_quote = True
                current_closing = quote_map[char]
                start_pos = i + 1  # 内容从左引号后开始
        else:
            if char == current_closing:
                content = prompt[start_pos:i].strip()
                if content:
                    results.append(content)
                in_quote = False
                current_closing = None
    
    return results


def parse_text_atlas(prompt):
    """改进版OCR文本提取（支持嵌套引号、转义字符、排除所有格）"""
    import re

    def recursive_extract(text):
        results = []
        quote_types = ["''", '""', "'", '"']  # 优先级：双单引号 > 双引号 > 单引号
        remaining_text = text
        
        for quote in quote_types:
            # 正则：排除所有格、支持转义字符、可选跳过'the text :'前缀
            pattern = (
                r'(?<!\w)'                     # 左侧非字母（排除所有格，如xxx's）
                r'(?:the\s+text\s*:\s*)?'      # 可选匹配'the text :'前缀
                r'(?<!\\)(?:\\\\)*'            # 允许转义字符（如 \", \'）
                + re.escape(quote) + 
                r'(.*?)'                       # 非贪婪匹配引号内内容
                r'(?<!\\)(?:\\\\)*'            # 允许转义字符
                + re.escape(quote) +
                r'(?!\w)'                      # 右侧非字母（排除所有格）
            )
            while True:
                match = re.search(pattern, remaining_text, re.DOTALL)
                if not match:
                    break
                start, end = match.span()
                content = match.group(1).strip()
                
                # 递归提取嵌套引号内容
                nested_results = recursive_extract(content)
                if nested_results:
                    results.extend(nested_results)
                else:
                    if content:  # 排除空内容
                        results.append(content)
                
                # 移除已匹配部分，避免重复提取
                remaining_text = remaining_text[:start] + remaining_text[end:]
        
        return results

    all_extracted = recursive_extract(prompt)
    # 去重（保留首次出现的内容）
    final_results = []
    seen = set()
    for text in all_extracted:
        if text not in seen:
            seen.add(text)
            final_results.append(text)
    
    return final_results


def parse_prompt(prompt, mode):
    """根据语言模式解析prompt（en：英文解析，cn/zh：中文解析）"""
    if mode in ['cn', 'zh']:
        ground_truth = parse_quoted_text(prompt)
        ground_truth = ' '.join(ground_truth)
    elif mode == 'en':
        ground_truth = parse_text_atlas(prompt)
        # 过滤所有引号后拼接
        filtered_truth = [text.replace('"', '').replace("'", '').replace('“', '').replace('”', '') 
                         for text in ground_truth if text]
        ground_truth = ' '.join(filtered_truth)
    else:
        raise ValueError(f'Unknown mode {mode}')
    return ground_truth
# 添加加载metadata函数
def load_meta_longtext(sample_dir, lang_mode,cache_dir):
    """加载metadata信息，根据数据集类型和语言模式处理"""

    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'longtext'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取对应的prompt文件路径
    prompt_file = META_DATA[dataset_type][lang_mode]
    
    # 检查文件是否存在
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
    
    # 加载prompt信息
    prompts = [json.loads(line) for line in open(prompt_file, 'r', encoding='utf-8')]
    prompt_map = {p['prompt_id']: p for p in prompts}
    
    data = []
    # 遍历样本目录中的所有图像文件
    for image_file in sorted(glob.glob(f'{sample_dir}/*.png')):
        fname = image_file.split('/')[-1]  # 获取文件名
        # 从文件名中提取prompt_id
        if '_' in fname:
            prompt_id = int(fname.split('_')[0])
        else:
            prompt_id = int(fname.split('.')[0])

    
        # 获取对应的prompt信息
        info = prompt_map[prompt_id]
        prompt = info['prompt']
        target = parse_prompt(prompt,mode=lang_mode)
        text = info['text']
        target_text = ' '.join(text)
        if len(target)==0:
            print(f'parse failed use ori target!')
            target = target_text
        if len(target)!=0 and (len(target_text)/len(target))>=2:
            print(f'parse failed for len << ori target use ori target')
            target = target_text
        # if target_text != target:
        #     print(f'extract target is {target}\n')
        #     print(f'while text is {target_text}')
        #     print(f'debug')
        data.append({
            'image': image_file,
            'prompt': prompt,
            'target': target,
            'ori_target':target_text,
            'id': f'{dataset_type}_{len(data)}'
        })
    
    return data
def load_meta_gentexteval(sample_dir, lang_mode,cache_dir):
    """加载metadata信息，根据数据集类型和语言模式处理"""

    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'gentexteval'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取对应的prompt文件路径
    prompt_file = META_DATA[dataset_type][lang_mode]
    
    # 检查文件是否存在
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
    
    # 加载prompt信息
    prompts = [json.loads(line) for line in open(prompt_file, 'r', encoding='utf-8')]
    prompt_map = {p['prompt_id']: p for p in prompts}

    
    data = []
    # 遍历样本目录中的所有图像文件
    for image_file in sorted(glob.glob(f'{sample_dir}/*.png')):
        fname = image_file.split('/')[-1]  # 获取文件名
        # 从文件名中提取prompt_id
        if '_' in fname:
            prompt_id = int(fname.split('_')[0])
        else:
            prompt_id = int(fname.split('.')[0])

    
        # 获取对应的prompt信息
        info = prompt_map[prompt_id]
        # print(info.keys())
        prompt = info['prompt']
        target = info['target']
        # target = parse_prompt(prompt,mode=lang_mode)
        # text = info['ori_target']
        # target_text = ' '.join(text)
        # target_text = info['ori_target']
        # if len(target)==0:
        #     print(f'parse failed use ori target!')
        #     target = target_text
        # if len(target)!=0 and (len(target_text)/len(target))>=2:
        #     print(f'parse failed for len << ori target use ori target')
        #     target = target_text
        # if target_text != target:
        #     print(f'extract target is {target}\n')
        #     print(f'while text is {target_text}')
        #     print(f'debug')
        data.append({
            'image': image_file,
            'prompt': prompt,
            'target': target,
            'ori_target':target,
            'id': f'{dataset_type}_{len(data)}'
        })
    
    return data
def load_meta_lex(sample_dir, lang_mode, cache_dir):
    """加载LeX数据集的metadata信息，支持easy、medium、hard三个子集"""
    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'lex'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取对应的prompt文件路径（现在是一个包含三个子集的字典）
    subset_files = META_DATA[dataset_type][lang_mode]
    
    data = []
    
    # 遍历三个子集（easy、medium、hard）
    for subset_type, prompt_file in subset_files.items():
        # 检查文件是否存在
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
        
        # 加载JSON文件
        subset_data = load_json(prompt_file)
        
        # 创建prompt映射（使用id字段）
        prompt_map = {p['id']: p for p in subset_data}
        
        # 构建当前子集的图像目录路径
        subset_image_dir = os.path.join(sample_dir, subset_type)
        
        # 遍历当前子集目录中的所有图像文件
        for image_file in sorted(glob.glob(f'{subset_image_dir}/*.png')):
            fname = os.path.basename(image_file)  # 获取文件名
            # 从文件名中提取id
            if '_' in fname:
                img_id = fname.split('_')[0]
            else:
                img_id = os.path.splitext(fname)[0]
        
            # 获取对应的prompt信息
            if img_id not in prompt_map:
                print(f"Warning: Could not find prompt for image {image_file} with id {img_id}, skipping...")
                continue
                
            info = prompt_map[img_id]
            prompt = info['enhanced_caption']
            
            # 使用text数组的内容通过空格连接成字符串作为target
            text_list = info['text']
            target = ' '.join(text_list)
        
            data.append({
                'image': image_file,
                'prompt': prompt,
                'target': target,
                'ori_target': target,
                'id': f'{dataset_type}_{len(data)}',
                'type': subset_type  # 添加type字段指示子集类型
            })
    
    return data
def load_meta_atlas(sample_dir, lang_mode, cache_dir):
    """加载atlasEval数据集的metadata信息，支持的四个子集"""
    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'atlas'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    subset_files = META_DATA[dataset_type][lang_mode]
    
    data = []
    
    # 遍历三个子集（easy、medium、hard）
    for subset_type, prompt_file in subset_files.items():
        # 检查文件是否存在
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
        
        # 加载JSON文件
        subset_data = load_json(prompt_file)
        
        # 创建prompt映射（使用id字段）
        prompt_map = {p['id']: p for p in subset_data}
        
        # 构建当前子集的图像目录路径
        subset_image_dir = os.path.join(sample_dir, subset_type)
        
        # 遍历当前子集目录中的所有图像文件
        for image_file in sorted(glob.glob(f'{subset_image_dir}/*.png')):
            fname = os.path.basename(image_file)  # 获取文件名
            # 从文件名中提取id
            if '_' in fname:
                img_id = fname.split('_')[0]
            else:
                img_id = os.path.splitext(fname)[0]
        
            # 获取对应的prompt信息
            if img_id not in prompt_map:
                print(f"Warning: Could not find prompt for image {image_file} with id {img_id}, skipping...")
                continue
                
            info = prompt_map[img_id]
            prompt = info['annotation']
            target = info['raw_text']
        
            data.append({
                'image': image_file,
                'prompt': prompt,
                'target': target,
                'ori_target': target,
                'id': f'{dataset_type}_{len(data)}',
                'type': subset_type  # 添加type字段指示子集类型
            })
    
    return data
def load_meta_tiif(sample_dir, lang_mode, cache_dir):
    
    # 根据不同的数据集类型使用不同的处理逻辑
    dataset_type = 'tiif'
    
    if lang_mode not in META_DATA[dataset_type]:
        raise ValueError(f"Language mode '{lang_mode}' not defined for dataset type '{dataset_type}' in META_DATA")
    
    # 获取对应的prompt文件路径
    subset_files = META_DATA[dataset_type][lang_mode]
    
    data = []
    
    # 遍历所有子集
    for subset_type, prompt_file in subset_files.items():
        # 检查文件是否存在
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Could not find prompt file at {prompt_file}")
        
        # 加载JSONL文件（已修改为使用load_jsonl函数）
        subset_data = load_jsonl(prompt_file)
        
        # 创建prompt映射（使用索引作为id，与用户提供的示例一致）
        prompt_map = {}
        for idx, p in enumerate(subset_data):
            p['id'] = str(idx)  # 为每个item生成一个id（使用索引）
            prompt_map[str(idx)] = p
        
        # 构建当前子集的图像目录路径
        subset_image_dir = os.path.join(sample_dir, subset_type)
        
        # 遍历当前子集目录中的所有图像文件
        for image_file in sorted(glob.glob(f'{subset_image_dir}/*.png')):
            fname = os.path.basename(image_file)  # 获取文件名
            # 从文件名中提取id（与用户提供的示例一致）
            if '_' in fname:
                img_id = fname.split('_')[0]
            else:
                img_id = os.path.splitext(fname)[0]
        
            # 获取对应的prompt信息
            if img_id not in prompt_map:
                print(f"Warning: Could not find prompt for image {image_file} with id {img_id}, skipping...")
                continue
                
            info = prompt_map[img_id]
            prompt = info['long_description']
            
            # 使用parse_prompt函数处理prompt
            target = parse_prompt(prompt, mode=lang_mode)
        
            data.append({
                'image': image_file,
                'prompt': prompt,
                'target': target,
                'ori_target': target,
                'id': f'{dataset_type}_{len(data)}',
                'type': subset_type  # 添加type字段指示子集类型
            })
    
    return data

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
def eval_datasets(args: argparse.Namespace, lang_mode: str):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    cache_dir = f"tmp/tmp_{formatted_time}"
    os.makedirs(cache_dir, exist_ok=True)#for oneig split webp
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
     # 加载元数据
    if args.type=='longtext':
        load_fuc = load_meta_longtext 
    elif args.type=='cvtg':
        load_fuc = load_meta_cvtg
    elif args.type=='oneig':
        load_fuc = load_meta_oneig
    elif args.type=='gentexteval':
        load_fuc = load_meta_gentexteval
    elif args.type=='lex':
        load_fuc = load_meta_lex
    elif args.type=='atlas':
        load_fuc = load_meta_atlas
    elif args.type=='tiif':
        load_fuc = load_meta_tiif
    else:
        print(f'no pair for {args.type}')
    try:
        data = load_fuc(args.sample_dir,lang_mode,cache_dir)
        print(f"Loaded {len(data)} samples for evaluation")
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return
    if not data:
        print("No samples found for evaluation")
        return
    result_jsonl_path = jsonl_path = os.path.join(output_dir, 'eval_results.jsonl')
    if os.path.exists(result_jsonl_path):
        # print(f"Warning: {result_jsonl_path} already exists. It will be overwritten.")
        print(f'{result_jsonl_path} already exists. Skip!')
        return 
    eval_data = gentext_eval(data)
    save(eval_data, output_dir)
    # if os.path.exists(cache_dir):
    #     shutil.rmtree(cache_dir, onerror=on_rm_error)
    


# 确保multi_score函数保持简单，直接返回分数字典
def multi_score(device, vllm_host="fdbd:dc03:9:393::36", vllm_port=8848):
    score_func_list = [
        # 'ocr_mtl',
        'textpecker_score_vllm']
    score_functions = {
        # "ocr_mtl": ocr_score_mtl,
        'textpecker_score_vllm': textpecker_score_vllm
    }
    score_fns = {}
    for score_name in score_func_list:
        if score_name == 'ocr_mtl':
            score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()
        elif score_name == 'textpecker_score_vllm':
            score_fns[score_name] = textpecker_score_vllm(vllm_host, vllm_port)  # 保持您的参数顺序

    # 直接返回原始分数字典，不计算加权平均
    def _fn(images, prompts, targets):
        score_details = {}
        
        # OCR评分
        # ocr_scores,ppocr_rec = score_fns['ocr_mtl'](images, prompts, targets)
        # score_details['ocr_scores'] = ocr_scores
        # score_details['ppocr_recs'] = ppocr_rec
        
        # TextPecker评分
        qua_scores, gned_scores, recs = score_fns['textpecker_score_vllm'](images, prompts, targets)
        score_details['pecker_quas'] = qua_scores
        score_details['pecker_gned'] = gned_scores
        score_details['pecker_recs'] = recs
        print(f'pecker_qua {qua_scores} pecker_gned {gned_scores} pecker_recs {recs}')
  
        
        return score_details
        
    return _fn


def gentext_eval(data, batch_size=8):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 初始化评分函数
    scorer = multi_score(device, VLLM_HOST, PORT)
    
    # 创建线程池
    executor = futures.ThreadPoolExecutor(max_workers=8)
    
    # 异步获取分数
    new_data = []
    start_time = time.time()
    
    # 计算总批次数量
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    print(f"Total batches to process: {total_batches} (batch size: {batch_size})")  # 新增：打印总批次信息
    
    # 使用tqdm创建进度条
    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, len(data), batch_size):
            # 计算当前批次号（从1开始）
            current_batch = i // batch_size + 1
            # 新增：打印当前批次进度（核心需求）
            print(f"Starting to process batch {current_batch}/{total_batches} (processing samples {i+1} to {min(i+batch_size, len(data))})")
            
            batch = data[i:i+batch_size]
            batch_start_time = time.time()
            
            try:
                # 准备批次数据 - 直接保持PIL图像格式
                pil_images = []
                prompts = []
                targets = []
                original_items = []
                
                for item in batch:
                    # 加载图像
                    try:
                        img = Image.open(item['image']).convert('RGB')
                        pil_images.append(img)
                        prompts.append(item['prompt'])
                        targets.append(item['target'])
                        original_items.append(item)
                    except Exception as e:
                        print(f"Failed to load image {item['image']}: {e}")
                        continue
                
                if not pil_images:
                    print(f"Batch {current_batch}/{total_batches}: No valid images, skipping")  # 新增：打印空批次信息
                    pbar.update(1)  # 更新进度条
                    continue
                
                # 异步提交任务获取rewards - 直接传递PIL图像
                future = executor.submit(scorer, pil_images, prompts, targets)
                time.sleep(0)  # 让出一些CPU时间
                
                # 获取异步任务的结果
                score_details = future.result()
                
                # 提取三个分数列表
                # ocr_scores = score_details['ocr_scores']
                pecker_quas = score_details['pecker_quas']
                pecker_gned = score_details['pecker_gned']
                
                # 确保三个分数列表等长，否则进行debug
                # if len(ocr_scores) != len(pecker_quas) or len(ocr_scores) != len(pecker_gned):
                #     print("Debug: 分数列表长度不匹配!")
                #     print(f"ocr_scores长度: {len(ocr_scores)}")
                #     print(f"pecker_quas长度: {len(pecker_quas)}")
                #     print(f"原始项目数量: {len(original_items)}")
                #     print(f"PIL图像数量: {len(pil_images)}")
                #     print(f"提示数量: {len(prompts)}")
                #     print(f"目标数量: {len(targets)}")

                # 整理结果，将分数添加到原始数据中
                for j, original_item in enumerate(original_items):
                    # 创建一个新的字典，包含原始数据和分数
                    result_item = original_item.copy()
                    
                    # 直接添加分数（假设数据总是完整的）
                    # result_item['ocr_score'] = ocr_scores[j]
                    result_item['pecker_qua'] = pecker_quas[j]
                    result_item['pecker_gned'] = pecker_gned[j]
                    result_item['pecker_recs'] = score_details['pecker_recs'][j]
                    # result_item['ppocr_recs'] = score_details['ppocr_recs'][j]
                    # print(f"""ocr_score: {ocr_scores[j]}, pecker_sem: {pecker_sems[j]}, pecker_qua: {pecker_quas[j]} 
                    #     pecker_gned: {result_item["pecker_gned"]} \n ppocr_recs: {result_item["ppocr_recs"]} pecker_recs: {result_item["pecker_recs"]}""")    
                    
                    new_data.append(result_item)

                
                batch_time = time.time() - batch_start_time
                # 新增：打印批次完成信息（含耗时）
                # print(f"Batch {current_batch}/{total_batches} completed in {batch_time:.2f}s (processed {len(original_items)} samples)")
                
                # 更新进度条并显示批次信息
                pbar.set_postfix(batch_time=f"{batch_time:.2f}s", 
                                current_batch=f"{current_batch}/{total_batches}")
                pbar.update(1)
                
            except Exception as e:
                # 新增：打印批次错误信息
                print(f"Error processing batch {current_batch}/{total_batches}: {e}")
                pbar.update(1)  # 即使出错也更新进度条
                continue
                
    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f}s. Processed {len(new_data)} samples.")
    
    return new_data
    
def save(new_data, output_dir):
    """将评估结果保存为jsonl格式"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果为jsonl文件
    jsonl_path = os.path.join(output_dir, 'eval_results.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in new_data:
            # 直接写入JSONL格式，假设数据已经是可序列化的
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {jsonl_path}")


def main():
    global PORT, VLLM_HOST
     # 解析命令行参数
    parser = argparse.ArgumentParser(description='TextPecker Evaluation Server')
    # 添加默认值以便于调试
    parser.add_argument('--sample_dir', type=str, 
                        default='eval/GenTextEval/eval_results/Qwenimage_ZH', 
                        help='Directory containing sample images')
    parser.add_argument('--output_dir', type=str, 
                        default='eval/TextPecker_eval/results/gentexteval/Qwenimage_ZH', 
                        help='Output directory for results')
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'en', 'zh'],
                        help='Language mode: auto (detect from model name), en, or zh')
    parser.add_argument('--port', type=int, default=8849, 
                        help='Port to run the server on (auto for random)')
    parser.add_argument('--vllm_host', type=str, default='localhost',
                        help='Host of the vllm server')
    parser.add_argument('--type', type=str, default='gentexteval', choices=['longtext', 'cvtg', 'oneig','gentexteval', 'lex', 'atlas', 'tiif'],
                        help='Dataset type to evaluate')
    args = parser.parse_args()
    
    PORT = args.port
    VLLM_HOST = args.vllm_host
    print(f'PORT IS {PORT}')
    print(f'VLLM_HOST IS {VLLM_HOST}')
    
    # 解析输入的模型名称（不包含后缀）
    base_sample_dir = args.sample_dir
    base_output_dir = args.output_dir
    
    # 根据mode和数据集类型确定需要评估的语言版本
    if args.mode == 'auto':
        # 对于auto模式，检测是否存在EN和ZH文件夹
        languages_to_evaluate = []
        
        # 检测EN文件夹
        en_sample_dir = base_sample_dir +"_EN"
        print(en_sample_dir)
        en_output_dir = base_output_dir +"_EN"
        if os.path.exists(en_sample_dir):
            languages_to_evaluate.append(('en', en_sample_dir, en_output_dir))
        
        # 检测ZH文件夹
        zh_sample_dir = base_sample_dir +"_ZH"
        zh_output_dir = base_output_dir +"_ZH"
        print(zh_sample_dir)
        if os.path.exists(zh_sample_dir):
            languages_to_evaluate.append(('zh', zh_sample_dir, zh_output_dir))
        print(f"Auto-detected {len(languages_to_evaluate)} language versions to evaluate")
        
    elif args.mode == 'en':
        # 对于en模式，首先尝试原始路径，然后尝试添加_EN后缀
        languages_to_evaluate = []
        
        # 首先尝试原始路径
        if os.path.exists(base_sample_dir):
            languages_to_evaluate.append(('en', base_sample_dir, base_output_dir))
        else:
            # 尝试添加_EN后缀
            en_sample_dir = base_sample_dir + "_EN"
            en_output_dir = base_output_dir +"_EN"
            if os.path.exists(en_sample_dir):
                languages_to_evaluate.append(('en', en_sample_dir, en_output_dir))
        
        if not languages_to_evaluate:
            print(f"Error: Could not find sample directory for EN mode: {base_sample_dir} or {en_sample_dir}")
            return
        
    elif args.mode == 'zh':
        # 对于zh模式，只尝试添加_ZH后缀
        zh_sample_dir = base_sample_dir +"_ZH"
        zh_output_dir = base_output_dir +"_ZH"
        
        if os.path.exists(zh_sample_dir):
            languages_to_evaluate = [('zh', zh_sample_dir, zh_output_dir)]
        else:
            print(f"Error: Could not find sample directory for ZH mode: {zh_sample_dir}")
            return
    
    # 执行多语言评估
    for lang_mode, sample_dir, output_dir in languages_to_evaluate:
        print(f"\n=== Evaluating {args.type} dataset for {lang_mode} language ===")
        print(f"Sample directory: {sample_dir}")
        print(f"Output directory: {output_dir}")
        
        # 创建临时参数对象
        temp_args = argparse.Namespace(**vars(args))
        temp_args.sample_dir = sample_dir
        temp_args.output_dir = output_dir
        
        # 执行评估
        eval_datasets(temp_args, lang_mode)
        print(f"=== Finished evaluation for {lang_mode} language ===\n")


if __name__ == "__main__":
    main()