import asyncio
from openai import AsyncOpenAI
from PIL import Image
import base64
from io import BytesIO
import os
from typing import Union
import random
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
textpecker_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(textpecker_root)
from parse_utils_pecker import get_score_v2, process_box_input, get_template


# --- 1. VLLM Server Configuration ---
PORT = 8848
VLLM_HOST = "2605:340:cd51:4b00:9181:c6bd:19c3:330" #replace with your server IP ##2605:340:cd60:0:9e1c:a6fd:1ee2:d01b
VLLM_BASE_URL = f"http://[{VLLM_HOST}]:{PORT}/v1" # Brackets are needed for IPv6 addresses in URLs
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "TextPecker"

# Initialize the client.
client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

def process_response(response, target):
    """处理单个响应并返回评分结果"""
    try:
        return get_score_v2(response, target, qua_amplify_factor=1.0, vis_cls=True)
    except Exception as e:
        print(f'error: {e}\ncur response:\n{response}')
        return 'ERROR', 'ERROR', {'error': f'{e}\n occur response:\n{response}'}

def textpecker_postprocess(response, target=None):
    """
    响应后处理：提取识别文本，计算质量分、语义分
    Args:
        response: LLMClient返回的原始响应
        target: 参考文本（可选）
    Returns:
        result: 包含识别结果、评分的字典
    """
    outputs = []
    for raw_output in response:
        quality_score, gned_score, cls_results = process_response(raw_output, target)
        output = {
        'qua_scores': quality_score,
        'sem_scores': gned_score,
        'full_results': cls_results
        }
        outputs.append(output)

    return outputs

# --- 2. Helper Functions (copied from rewards.py) ---
def pil_image_to_base64(image: Union[Image.Image, str]) -> str:
    """Converts a PIL image or image path to a base64 string for API calls."""
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image path not found: {image}")
        image = Image.open(image)
    buffered = BytesIO()
    image.convert("RGB").save(buffered, format="JPEG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_image_text}"

def get_template(ori_bbox=None):
    if ori_bbox is not None:
        PROMPT = f'''
            This is a text-generated image. Please recognize all visible text in the local area "{ori_bbox}". 
            Marking rules:
            1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
            2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
            Output in the following JSON format:
            {{
            "recognized_text": "Text in {ori_bbox} (including structural error markers)"
            }}
            '''
    else:
        PROMPT = """
                This is a text-generated image. Please recognize all visible text in the entire image.
                Marking rules: 
                1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
                2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
                Output in the following JSON format:
                {
                "recognized_text": "All text in the image (including structural error markers)"
                }
                """
    return PROMPT

# --- 3. Async API Call Functions (copied from rewards.py) ---
async def evaluate_image(prompt: str, base64_image: str):
    """Sends a single asynchronous request to the vLLM server."""
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
        return f"Request failed: {e}"

async def evaluate_batch_image(prompts: list[str], images: list[Union[Image.Image, str]]):
    """Creates and runs a batch of asynchronous evaluation tasks."""
    try:
        images_base64 = [pil_image_to_base64(img) for img in images]
        tasks = [evaluate_image(prompt, img_b64) for prompt, img_b64 in zip(prompts, images_base64)]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        return [f"Batch evaluation failed: {e}"] * len(prompts)

# --- 4. Main Execution Logic ---
async def main():
    """
    Main function to run the test. This simulates the call from the reward function.
    """
    print("Starting vLLM call test...")
    # Simple Recognition Demo
    # --- Configure test data (same as reward function context) ---
    EXAMPLES =[
        os.path.join(textpecker_root, 'examples', 'example0.jpg'),
        # os.path.join(textpecker_root, 'examples', 'example1.png'),
        # os.path.join(textpecker_root, 'examples', 'example2.png')
    ]
    IMAGE_PATH = random.choice(EXAMPLES)


    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file not found at '{IMAGE_PATH}'. Please check the path.")
        return


    repeate_nums = 2


    # 示例1：RFI模式 - 无bbox，无target
    print("=== 示例1：RFI模式 - 无bbox，无target ===")
    print("Sending request to vLLM server...")
    images = [IMAGE_PATH]*repeate_nums 
    vllm_prompts = [get_template()]*repeate_nums
    results = await evaluate_batch_image(vllm_prompts, images)
    postprocess_results = textpecker_postprocess(results, None)
    print(f"识别结果: {postprocess_results[0]['full_results']['recognized_text']}")
    print(f"质量分数: {postprocess_results[0]['qua_scores']}")
    print(f"语义分数: {postprocess_results[0]['sem_scores']}\n")
    
    # 示例2：RGI模式 - 无bbox，有target
    print("\n=== 示例2：RGI模式 - 无bbox，有target ===")
    target = '《去噪扩散概率模型》\n作者名单：Jonathan Ho, Ajay Jain, Pieter Abbeel\n摘要：我们利用扩散概率模型（一类受非平衡热力学考虑启发的潜变量模型）呈现了高质量的图像合成结果。通过在一个加权变分界限上进行训练（该界限是根据扩散概率模型与Langevin动态去噪得分匹配之间的新颖连接设计的），我们获得了最佳的结果。我们的模型自然地采用一种渐进的有损解压方案，可以被解释为自回归解码的推广。在无条件CIFAR10数据集上，我们得到9.46的Inception得分和3.17的最新FID得分。在$256 \times 256$的LSUN数据集上，我们获得了与ProgressiveGAN相似的样本质量。我们的实现可以在https://github.com/hojonathanho/diffusion找到。\n 1.介绍 \n各种深度生成模型最近在多种数据模式中展示了高质量的样本。生成对抗网络（GAN）、自回归模型、流模型和变分自编码器（VAE）已合成了引人注目的图像和音频样本[14, 27, 3, $58,38,25,10,32,44,57,26,33,45]$，同时，在能量模型和得分匹配方面也有了显著进展，生成了与GAN相媲美的图像[11,55]。'
    postprocess_results_with_target = textpecker_postprocess(results, target)
    print(f"识别结果: {postprocess_results_with_target[0]['full_results']['recognized_text']}")
    print(f"质量分数: {postprocess_results_with_target[0]['qua_scores']}")
    print(f"语义分数: {postprocess_results_with_target[0]['sem_scores']}\n")
    
    # 示例3：RFB模式 - 有bbox，无target
    print("\n=== 示例3：RFB模式 - 有bbox，无target ===")
    bbox_px = [310, 30, 730, 90]
    # 先进行box预处理
    normalized_bbox = process_box_input(bbox_px, IMAGE_PATH)
    bbox_prompts = [get_template(normalized_bbox)] * repeate_nums
    bbox_results = await evaluate_batch_image(bbox_prompts, images)
    bbox_postprocess_results = textpecker_postprocess(bbox_results, None)
    print(f"识别结果: {bbox_postprocess_results[0]['full_results']['recognized_text']}")
    print(f"质量分数: {bbox_postprocess_results[0]['qua_scores']}")
    print(f"语义分数: {bbox_postprocess_results[0]['sem_scores']}\n")
    
    # 示例4：RGB模式 - 有bbox，有target
    print("\n=== 示例4：RGB模式 - 有bbox，有target ===")
    bbox_target = '《去噪扩散概率模型》、'
    bbox_target_results = textpecker_postprocess(bbox_results, bbox_target)
    print(f"识别结果: {bbox_target_results[0]['full_results']['recognized_text']}")
    print(f"质量分数: {bbox_target_results[0]['qua_scores']}")
    print(f"语义分数: {bbox_target_results[0]['sem_scores']}\n")

   

    print("Demo finished.")


if __name__ == "__main__":
    asyncio.run(main())