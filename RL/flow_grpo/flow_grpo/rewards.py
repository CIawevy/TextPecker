from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict
from flow_grpo.parse_utils_pecker import get_score_v2
from typing import Union
import time
import threading
def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score():
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn
def textpecker_score_local(device):
    #contains bug
    from flow_grpo.qwenvl import TextPeckerScorer

    scorer =TextPeckerScorer(dtype=torch.bfloat16, device=device)
    def contain_chinese(prompts):
        return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False
    def _fn(images, prompts, metadata,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        targets = [target for target in targets]
        # modes = ['cn' if contain_chinese(prompt) else 'en' for prompt in prompts]
        scores = scorer(prompts, images,targets)
        return scores, {}

    return _fn
    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn
def ocr_score_cn(device):
    from flow_grpo.ocr import OcrScorerCN

    scorer = OcrScorerCN()

    def _fn(images, prompts, metadata,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts,targets)
        # change tensor to list
        return scores, {}

    return _fn
def ocr_score_en(device):
    from flow_grpo.ocr import OcrScorerEN

    scorer = OcrScorerEN()

    def _fn(images, prompts, metadata,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts,targets)
        # change tensor to list
        return scores, {}

    return _fn
def ocr_score_mtl(device):
    #中英文的ocr score function
    from flow_grpo.ocr import OcrScorerEN
    from flow_grpo.ocr import OcrScorerCN

    en_scorer = OcrScorerEN()
    cn_scorer = OcrScorerCN()
    def contain_chinese(prompts):
        return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False

    def _fn(images, prompts, metadata,targets):
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
        
        # 中文组推理
        if cn_indices:
            cn_images = [images[i] for i in cn_indices]
            cn_prompts = [prompts[i] for i in cn_indices]
            cn_targets = [targets[i] for i in cn_indices]
            cn_scores= cn_scorer(cn_images, cn_prompts,cn_targets)
            for i, idx in enumerate(cn_indices):
                scores[idx] = cn_scores[i]
        
        # 英文组推理
        if en_indices:
            en_images =  [images[i] for i in en_indices]
            en_prompts = [prompts[i] for i in en_indices]
            en_targets = [targets[i] for i in en_indices]
            en_scores = en_scorer(en_images, en_prompts,en_targets)
            for i, idx in enumerate(en_indices):
                scores[idx] = en_scores[i]
        
  
        return scores, {}

    return _fn
def ocr_score_gned_cn(device):
    from flow_grpo.ocr import OcrScorerCN

    scorer = OcrScorerCN()

    def _fn(images, prompts, metadata,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts,targets,use_gned=True)
        # change tensor to list
        return scores, {}

    return _fn
def ocr_score_gned_en(device):
    from flow_grpo.ocr import OcrScorerEN

    scorer = OcrScorerEN()

    def _fn(images, prompts, metadata,targets):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts,targets,use_gned=True)
        # change tensor to list
        return scores, {}

    return _fn
def ocr_score_gned_mtl(device):
    #中英文的ocr score function
    from flow_grpo.ocr import OcrScorerEN
    from flow_grpo.ocr import OcrScorerCN

    en_scorer = OcrScorerEN()
    cn_scorer = OcrScorerCN()
    def contain_chinese(prompts):
        return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False

    def _fn(images, prompts, metadata,targets):
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
        
        # 中文组推理
        if cn_indices:
            cn_images = [images[i] for i in cn_indices]
            cn_prompts = [prompts[i] for i in cn_indices]
            cn_targets = [targets[i] for i in cn_indices]
            cn_scores= cn_scorer(cn_images, cn_prompts,cn_targets,use_gned=True)
            for i, idx in enumerate(cn_indices):
                scores[idx] = cn_scores[i]
        
        # 英文组推理
        if en_indices:
            en_images =  [images[i] for i in en_indices]
            en_prompts = [prompts[i] for i in en_indices]
            en_targets = [targets[i] for i in en_indices]
            en_scores = en_scorer(en_images, en_prompts,en_targets,use_gned=True)
            for i, idx in enumerate(en_indices):
                scores[idx] = en_scores[i]
        
  
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://127.0.0.1:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.82.120.15:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn
# 首先在文件顶部添加线程本地存储对象和客户端缓存
_thread_local = threading.local()
_process_clients = {}
_vllm_client_lock = threading.Lock()

# 修改textpecker_score_vllm函数
def textpecker_score_vllm_eval(device, vllm_host="fdbd:dc03:9:393::36", vllm_port=8848):
    """
    Computes an OCR-based reward score using a remote TextPecker model served with vLLM.
    This version is aligned with the local TextPeckerScorer logic from qwenvl.py,
    calculating a weighted score from semantic and quality components.
    """
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import torch
    # This import is based on the user's provided file `qwenvl.py`
    # from flow_grpo.parse_utils_pecker import get_score

    # --- 1. VLLM Server Configuration ---  
    VLLM_HOST = vllm_host
    PORT = vllm_port
    # Format URL for IPv6 if needed
    if ":" in VLLM_HOST:
        VLLM_BASE_URL = f"http://[{VLLM_HOST}]:{PORT}/v1"
    else:
        VLLM_BASE_URL = f"http://{VLLM_HOST}:{PORT}/v1"
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

    def _fn(images, prompts, metadata, targets):
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
        rewards = []
        scaled_qua_scores =[]
        semantic_scores = []
        qua_w,sem_w = 0.5,0.5
        qua_amplify_factor = 1.0 # 1/5 = 20% 
        for j,response in enumerate(text_outputs):
            try:
                # sem_score, qua_score, gned_score,rec_text, correct_text = get_score('RGI', response, targets[j])
                scaled_quality_score,semantic_score,cls_results = get_score_v2(response, targets[j],qua_amplify_factor)
            except Exception as e:
                print(f'error:{e}')
                print(f'cur response:\n{response}')
                scaled_quality_score,semantic_score,cls_results = 0, 0, {}
            final_score = (qua_w * scaled_quality_score + sem_w * semantic_score)/ (sem_w+qua_w)  #weigted sum of two score
            rewards.append(final_score)
            scaled_qua_scores.append(scaled_quality_score)
            semantic_scores.append(semantic_score)
        
        return  rewards, scaled_qua_scores, semantic_scores

    return _fn
# 首先在文件顶部添加线程本地存储对象和客户端缓存
_thread_local = threading.local()
_process_clients = {}
_vllm_client_lock = threading.Lock()

# 修改textpecker_score_vllm函数
def textpecker_score_vllm_amp5(device, vllm_host="fdbd:dc03:9:393::36", vllm_port=8848):
    """
    Computes an OCR-based reward score using a remote TextPecker model served with vLLM.
    This version is aligned with the local TextPeckerScorer logic from qwenvl.py,
    calculating a weighted score from semantic and quality components.
    """
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import torch
    # This import is based on the user's provided file `qwenvl.py`
    # from flow_grpo.parse_utils_pecker import get_score

    # --- 1. VLLM Server Configuration ---  
    VLLM_HOST = vllm_host
    PORT = vllm_port
    # Format URL for IPv6 if needed
    if ":" in VLLM_HOST:
        VLLM_BASE_URL = f"http://[{VLLM_HOST}]:{PORT}/v1"
    else:
        VLLM_BASE_URL = f"http://{VLLM_HOST}:{PORT}/v1"
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

    def _fn(images, prompts, metadata, targets):
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
        rewards = []
        scaled_qua_scores =[]
        semantic_scores = []
        qua_w,sem_w = 0.5,0.5
        qua_amplify_factor = 5.0 # 1/5 = 20% 
        for j,response in enumerate(text_outputs):
            try:
                # sem_score, qua_score, gned_score,rec_text, correct_text = get_score('RGI', response, targets[j])
                scaled_quality_score,semantic_score,cls_results = get_score_v2(response, targets[j],qua_amplify_factor)
            except Exception as e:
                print(f'error:{e}')
                print(f'cur response:\n{response}')
                scaled_quality_score,semantic_score,cls_results = 0, 0, {}
            final_score = (qua_w * scaled_quality_score + sem_w * semantic_score)/ (sem_w+qua_w)  #weigted sum of two score
            rewards.append(final_score)
            scaled_qua_scores.append(scaled_quality_score)
            semantic_scores.append(semantic_score)
        
        return  rewards, scaled_qua_scores, semantic_scores

    return _fn

def textpecker_score_vllm(device, vllm_host="fdbd:dc03:9:393::36", vllm_port=8848):
    """
    Computes an OCR-based reward score using a remote TextPecker model served with vLLM.
    This version is aligned with the local TextPeckerScorer logic from qwenvl.py,
    calculating a weighted score from semantic and quality components.
    """
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import torch
    # This import is based on the user's provided file `qwenvl.py`
    # from flow_grpo.parse_utils_pecker import get_score

    # --- 1. VLLM Server Configuration ---  
    VLLM_HOST = vllm_host
    PORT = vllm_port
    # Format URL for IPv6 if needed
    if ":" in VLLM_HOST:
        VLLM_BASE_URL = f"http://[{VLLM_HOST}]:{PORT}/v1"
    else:
        VLLM_BASE_URL = f"http://{VLLM_HOST}:{PORT}/v1"
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
                    temperature=0,
                    max_tokens=2048,
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

    def _fn(images, prompts, metadata, targets):
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
        rewards = []
        scaled_qua_scores =[]
        semantic_scores = []
        qua_w,sem_w = 0.0,1.0
        qua_amplify_factor = 5.0 # 1/5 = 20% 
        for j,response in enumerate(text_outputs):
            try:
                # sem_score, qua_score, gned_score,rec_text, correct_text = get_score('RGI', response, targets[j])
                scaled_quality_score,semantic_score,cls_results = get_score_v2(response, targets[j],qua_amplify_factor)
            except Exception as e:
                print(f'error:{e}')
                print(f'cur response:\n{response}')
                scaled_quality_score,semantic_score,cls_results = 0, 0, {}
            final_score = (qua_w * scaled_quality_score + sem_w * semantic_score)/ (sem_w+qua_w)  #weigted sum of two score
            rewards.append(final_score)
            scaled_qua_scores.append(scaled_quality_score)
            semantic_scores.append(semantic_score)
        
        return  rewards, scaled_qua_scores, semantic_scores

    return _fn
def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn
def multi_score_ori(device, score_dict):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        # "ocr_mtl": ocr_score_mtl,
        # "ocr_en": ocr_score_en,
        # 'ocr_cn':ocr_score_cn,
        # "textpecker_score": TextPecker_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                #for ocr scores
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn
def multi_score(device, score_dict,vllm_host="fdbd:dc03:9:393::36", vllm_port=8848):
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score, #flow grpo ori ocr score
        "ocr_mtl": ocr_score_mtl, #ppocr + ocr score
        "ocr_en": ocr_score_en, #ppocr + ocr score
        'ocr_cn':ocr_score_cn, #ppocr + ocr score
        'ocr_gned_en':ocr_score_gned_en, #ppocr  + gned
        'ocr_gned_cn':ocr_score_gned_cn, #ppocr + gned
        'ocr_gned_mtl':ocr_score_gned_mtl, #ppocr + gned
        "textpecker_score": textpecker_score_local, #abandomed old version
        'textpecker_score_vllm': textpecker_score_vllm, #TextPecker + gned + scaled quality score
        'textpecker_score_vllm_amp5': textpecker_score_vllm_amp5, #TextPecker + gned + scaled quality score
        'textpecker_score_vllm_eval': textpecker_score_vllm_eval, #TextPecker + gned + scaled quality score
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns={}
    for score_name, weight in score_dict.items():
        if 'textpecker_score'  in score_name:
            score_fns[score_name] = score_functions[score_name](device,vllm_host,vllm_port)
        else:
            score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()


    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, targets, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        #TODO:multi reward 要修改
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
            elif score_name == "image_similarity":
                scores, rewards = score_fns[score_name](images, ref_images)
            elif  'textpecker_score' in score_name:
                scores, qua_scores, sem_scores = score_fns[score_name](images, prompts, metadata, targets)
                score_details['qua_scores'] = qua_scores
                score_details['sem_scores'] = sem_scores
            elif score_name == 'pickscore':
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            elif score_name == 'aesthetic':
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            else:
                #for ocr scores
                scores, rewards = score_fns[score_name](images, prompts, metadata, targets)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        score_details['avg'] = total_scores
        return score_details, {}
    return _fn

def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()