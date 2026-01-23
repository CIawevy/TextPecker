from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Tuple
from PIL import Image
import os
import re 

# def parse_quoted_text(prompt):
#     """
#     提取引号包裹的文本内容，支持中英文多种引号类型：
#     - 中文双引号：“...””
#     - 中文单引号：‘...’
#     - 英文双引号："..."
#     - 英文单引号：'...'
#     使用贪心算法从左到右匹配引号对，提取所有顶层引号内容
#     """
#     # 定义引号映射表：左引号 -> 右引号
#     quote_map = {
#         '“': '”',   # 中文双引号
#         '‘': '’',   # 中文单引号
#         '"': '"',   # 英文双引号
#         "'": "'",    # 英文单引号
#         '《': '》'   # 书名号
#     }
    
#     results = []
#     in_quote = False  # 引号内状态标记
#     current_closing = None  # 当前等待匹配的右引号
#     start_pos = 0  # 引号内容起始位置
    
#     for i, char in enumerate(prompt):
#         if not in_quote:
#             # 遇到左引号时进入引号内状态
#             if char in quote_map:
#                 in_quote = True
#                 current_closing = quote_map[char]
#                 start_pos = i + 1  # 内容从左引号后开始
#         else:
#             # 遇到对应右引号时提取内容
#             if char == current_closing:
#                 # 提取引号间内容并去除首尾空白
#                 content = prompt[start_pos:i].strip()
#                 if content:  # 忽略空内容
#                     results.append(content)
#                 # 重置引号状态
#                 in_quote = False
#                 current_closing = None
    
#     return results

# def parse_text_atlas(prompt):
#     """
#     改进版OCR文本提取函数，解决以下问题：
#     1. 支持单引号 ('...'), 双引号 ("..."), 以及连续两个单引号 (''...'')
#     2. 递归提取嵌套引号内的文本
#     3. 正确处理转义字符（如 \", \'）
#     4. 排除所有格形式（如 xxx's）
#     5. 避免重复提取
#     6. 跳过 'the text :' 前缀
#     """
#     import re

#     def recursive_extract(text):
#         results = []
#         # 定义不同引号类型
#         quote_types = ["''", '""', "'", '"']
#         remaining_text = text
#         for quote in quote_types:
#             # 动态生成正则表达式，排除所有格形式并跳过 'the text :' 前缀
#             pattern = (
#                 r'(?<!\w)'                     # 确保左侧不是字母（排除所有格）
#                 r'(?:the\s+text\s*:\s*)?'      # 可选的 'the text :' 前缀
#                 r'(?<!\\)(?:\\\\)*'            # 允许转义字符（如 \", \'）
#                 + re.escape(quote) + 
#                 r'(.*?)'                       # 非贪婪匹配内容
#                 r'(?<!\\)(?:\\\\)*'            # 允许转义字符
#                 + re.escape(quote) +
#                 r'(?!\w)'                      # 确保右侧不是字母（排除所有格）
#             )
#             while True:
#                 match = re.search(pattern, remaining_text, re.DOTALL)
#                 if not match:
#                     break
#                 start, end = match.span()
#                 content = match.group(1)
#                 # 递归提取嵌套内容
#                 nested_results = recursive_extract(content)
#                 if nested_results:
#                     results.extend(nested_results)
#                 else:
#                     results.append(content.strip())
#                 # 移除已匹配的部分
#                 remaining_text = remaining_text[:start] + remaining_text[end:]
#         return results

#     all_extracted = recursive_extract(prompt)
#     final_results = []
#     for text in all_extracted:
#         # 去重并保留最长有效文本
#         if text and text not in final_results:
#             final_results.append(text)
#     return final_results
# def parse_prompt(prompt,mode):
#     if mode=='cn':
#         ground_truth = parse_quoted_text(prompt)
#         ground_truth = ' '.join(ground_truth)
#     elif mode=='en':
#         ground_truth = parse_text_atlas(prompt)
#         # 过滤掉所有引号后再拼接
#         filtered_truth = []
#         for text in ground_truth:
#             # 移除所有类型的引号
#             filtered_text = text.replace('"', '').replace("'", '').replace('“', '').replace('”', '')
#             if filtered_text:  # 确保过滤后不为空
#                 filtered_truth.append(filtered_text)
#         ground_truth = ' '.join(filtered_truth)
#     else:
#         raise ValueError(f'Unknown mode {mode}')
#     return ground_truth
class OcrScorer:
    # def __init__(self, use_gpu: bool = False):
    #     """
    #     OCR reward calculator
    #     :param use_gpu: Whether to use GPU acceleration for PaddleOCR
    #     """
    #     self.ocr = PaddleOCR(
    #         use_angle_cls=False,
    #         lang="en",
    #         use_gpu=use_gpu,
    #         show_log=False  # Disable unnecessary log output
    #     )
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        :param custom_model_dir: Custom directory for PaddleOCR models
        """
        # 设置自定义模型目录
        # if custom_model_dir is None:
        # 默认使用项目目录下的models文件夹
        custom_model_dir = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/models/paddleocr_rewards"
        
        # 设置环境变量改变PaddleOCR的模型下载路径
        # os.environ["PADDLEOCR_HOME"] = custom_model_dir
        
        
        # 初始化PaddleOCR，指定模型路径
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,  # Disable unnecessary log output
            det_model_dir=os.path.join(custom_model_dir, "whl/det/en/en_PP-OCRv3_det_infer"),
            rec_model_dir=os.path.join(custom_model_dir, "whl/rec/en/en_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(custom_model_dir, "whl/rec/en/en_PP-OCRv4_rec_infer"),
            text_detection_model_name ='PP-OCRv3_det' ,
            text_recognition_model_name ='PP-OCRv4_rec',
        )
        

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
            rewards.append(reward)

        return rewards

class OcrScorerEN:
    # def __init__(self, use_gpu: bool = False):
    #     """
    #     OCR reward calculator
    #     :param use_gpu: Whether to use GPU acceleration for PaddleOCR
    #     """
    #     self.ocr = PaddleOCR(
    #         use_angle_cls=False,
    #         lang="en",
    #         use_gpu=use_gpu,
    #         show_log=False  # Disable unnecessary log output
    #     )
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        :param custom_model_dir: Custom directory for PaddleOCR models
        """
        # 设置自定义模型目录
        # if custom_model_dir is None:
        # 默认使用项目目录下的models文件夹
        custom_model_dir = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/models/paddleocr_rewards"
        
        # 设置环境变量改变PaddleOCR的模型下载路径
        # os.environ["PADDLEOCR_HOME"] = custom_model_dir
        
        
        # 初始化PaddleOCR，指定模型路径
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False,  # Disable unnecessary log output
            det_model_dir=os.path.join(custom_model_dir, "whl/det/en/en_PP-OCRv3_det_infer"),
            rec_model_dir=os.path.join(custom_model_dir, "whl/rec/en/en_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(custom_model_dir, "whl/cls/ch_ppocr_mobile_v2.0_cls_infer"),
            text_detection_model_name ='PP-OCRv3_det' ,
            text_recognition_model_name ='PP-OCRv4_rec',
        )
        

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str],
                targets: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        # # prompts = [prompt.split('"')[1] for prompt in prompts]
        # raw_prompts = prompts.copy()
        # # prompts = [parse_prompt(prompt,'en') for prompt in prompts]
        # parse_texts = []
        # for p in prompts:
        #     parse_text = parse_prompt(p,'en')
        #     if len(parse_text)==0:
        #         parse_text = parse_prompt(p,'cn')
        #         print(f'check now en:{parse_prompt(p,"en")}')
        #         print(f'check now cn:{parse_prompt(p,"cn")}')
        #         print(f'cur prompts:{raw_prompts}')
        #         print(f'replace with quote parse result')
        #     assert len(parse_text)>0,f'parse_text is empty pls check:{p} '
        #     parse_texts.append(parse_text)

       
       
        # prompts = [''.join(parse_text(prompt)) for prompt in prompts]
        recs = []
        rewards = []
        # Ensure input lengths are consistent
        assert len(targets)==len(prompts),f'len(targets)={len(targets)},len(prompts)={len(prompts)} targets:{targets}\n prompts:{prompts}'

        assert len(images) == len(targets), "Images and targets must have the same length"
        for img, prompt,p in zip(images, targets,prompts):
            if len(prompt)==0:
                print(f'batch targets :{targets}')
                # print(f'batch_prompts:{prompts}')
            assert len(prompt)>0,f'check'
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                rec = recognized_text
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f'{e}')
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
            rewards.append(reward)
            recs.append(rec)

        return rewards , recs

class OcrScorerCN:
    # def __init__(self, use_gpu: bool = False):
    #     """
    #     OCR reward calculator
    #     :param use_gpu: Whether to use GPU acceleration for PaddleOCR
    #     """
    #     self.ocr = PaddleOCR(
    #         use_angle_cls=False,
    #         lang="en",
    #         use_gpu=use_gpu,
    #         show_log=False  # Disable unnecessary log output
    #     )
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        :param custom_model_dir: Custom directory for PaddleOCR models
        """
        # 设置自定义模型目录
        # if custom_model_dir is None:
        # 默认使用项目目录下的models文件夹
        custom_model_dir = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/models/paddleocr_rewards"
        
        # 设置环境变量改变PaddleOCR的模型下载路径
        # os.environ["PADDLEOCR_HOME"] = custom_model_dir
        
        
        # 初始化PaddleOCR，指定模型路径
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            use_gpu=use_gpu,
            show_log=False,  # Disable unnecessary log output
            det_model_dir=os.path.join(custom_model_dir, "whl/det/ch/ch_PP-OCRv4_det_infer"),
            rec_model_dir=os.path.join(custom_model_dir, "whl/rec/ch/ch_PP-OCRv4_rec_infer"),
            cls_model_dir=os.path.join(custom_model_dir, "whl/cls/ch_ppocr_mobile_v2.0_cls_infer"),
            text_detection_model_name ='PP-OCRv4_det' ,
            text_recognition_model_name ='PP-OCRv4_rec',
        )
        

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str],
                targets: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        Clawer made cn ocr reward
        """
        # raw_prompts = prompts.copy()
        # parse_texts = []
        # for p in prompts:
        #     parse_text = parse_prompt(p,'cn')
        #     assert len(parse_text)>0,f'parse_text is empty pls check:{p} '
        #     parse_texts.append(parse_text)

        rewards = []
        recs = []
        # Ensure input lengths are consistent
        assert len(targets)==len(prompts),f'len(targets)={len(targets)},len(prompts)={len(prompts)}'
        assert len(images) == len(targets), "Images and targets must have the same length"
        for img, prompt in zip(images, targets):
            # if len(prompt)==0:
                # print(f'batch targets :{targets}')
                # print(f'batch_prompts:{prompts}')
            assert len(prompt)>0,f'target is empty'
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                 # 处理文本（移除空格，中文不需要lower()）
                rec = recognized_text
                recognized_text = recognized_text.replace(' ', '')
                prompt = prompt.replace(' ', '')
                
                # 计算匹配得分：遍历识别字符，在prompt中找到则+1并移除该字符
                prompt_list = list(prompt)
                score = 0
                for c in recognized_text:
                    if c in prompt_list:
                        score += 1
                        prompt_list.remove(c)  # 移除第一个匹配字符，避免重复匹配
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                recognized_text = ''
                score = 0

            # 计算奖励：得分除以两个文本长度的最大值
            max_len = max(len(prompt), len(recognized_text))
            reward = score / max_len if max_len != 0 else 0.0
            rewards.append(reward)
            recs.append(rec)

        return rewards, recs
class OcrScorer_video_or_image:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False  # Disable unnecessary log output
        )
        self.frame_interval = 4

    @torch.no_grad()
    def __call__(self, images: Union[List[Image.Image], List[np.ndarray]], prompts: List[str]) -> Tuple[List[float], torch.Tensor]:
        """
        :param images: List of images or videos (each video as np.ndarray of shape [F, H, W, C])
        :param prompts: List of prompts containing target text
        :return: (List of OCR rewards, Tensor of attention regions)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        assert len(images) == len(prompts), "Mismatch between images and prompts."

        rewards = []
        for img, prompt in zip(images, prompts):
            prompt = prompt.replace(' ', '').lower()
            frame_rewards = []

            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
            else:
                sampled_frames = [img]

            for frame in sampled_frames:
                region = None
                if isinstance(frame, Image.Image):
                    frame = np.array(frame)
                try:
                    result = self.ocr.ocr(frame, cls=False)
                    text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                    text = text.replace(' ', '').lower()

                    dist = distance(text, prompt)
                    dist = min(dist, len(prompt))
    
                except Exception as e:
                    print(f"OCR failed on frame: {e}")
                    dist = len(prompt)

                reward = 1 - dist / len(prompt)
                if reward > 0:
                    frame_rewards.append(reward)

            if frame_rewards:
                rewards.append(sum(frame_rewards) / len(frame_rewards))
            else:
                rewards.append(0.0)

        return rewards

# if __name__ == "__main__":
#     # example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
#     # example_image = Image.open(example_image_path)
#     # example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
#     # # Instantiate scorer
#     # scorer = OcrScorer(use_gpu=False)

#     # # Call scorer and print result
#     # reward = scorer([example_image], [example_prompt])
#     # print(f"OCR MLT Reward: {reward}")



#     example_image_path = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/advs_data/ads_images_gen/000a5d5b6311356a7a94aea84bbfd4f3_8726863281.jpg"
#     example_image = Image.open(example_image_path)
#     example_prompt = '这是一张中文文字生成图片，文字内容为"加厚加粗 钢管臂力器 狂练胸肌 告别拜拜肉",“有点离谱了”'
#     parsed_prompt = parse_quoted_text(example_prompt)
#     # example_prompt = '这是一张中文文字生成图片，文字内容为"加厚加粗 钢管臂力器 狂练胸肌 告别拜拜肉"'
#     # Instantiate scorer
#     # en_scorer = OcrScorerEN(use_gpu=False)
#     cn_scorer = OcrScorerCN(use_gpu=False)
  

#     # # if contain_chinese(example_prompt):
#     scores = cn_scorer([example_image], [example_prompt])
#     # else:
#         # scores = en_scorer([example_image], [example_prompt])
#     # Call scorer and print result
#     reward = scores
#     print(f"OCR Reward: {reward}")

    