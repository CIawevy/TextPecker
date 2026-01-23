import pandas as pd
import json
import os
from PIL import Image  # 新增图像操作库
import math  # 新增数学计算库
import re  # 新增正则表达式库
import numpy as np
from tqdm import tqdm
import json
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties  # 新增字体管理库
import random
import sys
sys.path.append('/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger') #replace with your own
# 添加中文字体设置
zhfont2 = FontProperties(fname="resources/SourceHanSansSC-Bold.otf") 
zhfont1 = FontProperties(fname='resources/SourceHanSansSC-Regular.otf')

import re
import textwrap
MIN_PIXELS = 768*28*28
MAX_PIXELS = 1024*28*28
IMAGE_FACTOR = 28
MAX_RATIO = 200
BOX_IDX = 0
IMG_IDX= 0 
bad_counter = 0
MIN_PIXELS = 768*28*28
MAX_PIXELS = 1024*28*28
IMAGE_FACTOR = 28
MAX_RATIO = 200
BOX_IDX = 0
IMG_IDX= 0 
bad_counter = 0

def round_by_factor(x, factor):
    return round(x / factor) * factor

def floor_by_factor(x, factor):
    return math.floor(x / factor) * factor

def ceil_by_factor(x, factor):
    return math.ceil(x / factor) * factor
def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
def convert_four_points_to_bbox(four_points):
    """
    将四个点的坐标转换为对角线两点的坐标
    :param four_points: 二维列表，包含四个点的坐标
    :return: 对角线两点的坐标 [x_min, y_min, x_max, y_max]
    """
    x_coords = [p[0] for p in four_points]
    y_coords = [p[1] for p in four_points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def resize_box(image_path,ocr_location,min_pixels,max_pixels):
    image = Image.open(image_path)
    original_width, original_height = image.size
    new_height, new_width = smart_resize(original_height, original_width,min_pixels=min_pixels,max_pixels=max_pixels)

    # 计算缩放比例
    height_ratio = new_height / original_height
    width_ratio = new_width / original_width

    # 将四个点的坐标转换为对角线两点的坐标
    # bbox = convert_four_points_to_bbox(ocr_location)
    bbox = ocr_location
    
    # 缩放 bbox 坐标
    scaled_bbox = [
        int(bbox[0] * width_ratio),
        int(bbox[1] * height_ratio),
        int(bbox[2] * width_ratio),
        int(bbox[3] * height_ratio)
    ]
    return scaled_bbox,bbox

def load_json_file(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
def load_jsonl_file(jsonl_file_path):
    """
    加载JSONL格式文件（每行一个JSON对象）
    
    参数：
        jsonl_file_path: JSONL文件的路径
    
    返回：
        list: 包含文件中所有JSON对象的列表
    """
    data = []
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 跳过空行
            line = line.strip()
            if not line:
                continue
            # 解析每行的JSON对象并添加到列表
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"解析JSONL文件时出错（行内容：{line}）：{e}")
    return data

def generate_qa_from_raw_data(raw_data_path,output_json_dir,cons_box_per_img):
    ROOT_DIR = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger'# 需替换为实际数据路径
    global bad_counter
    """从原始中间数据生成QA训练数据"""
    # raw_data_list = load_json_file(raw_data_path)
    raw_data_list = load_jsonl_file(raw_data_path)
    
    img_qa_list = []
    box_qa_list = []
    for current_item in tqdm(raw_data_list, desc="生成QA数据"):
        img_qa_samples, box_qa_samples = process_data(current_item,cons_box_per_img,ROOT_DIR)
        box_qa_list.extend(box_qa_samples)
        img_qa_list.extend(img_qa_samples)
    #随机抽取50000个样本
    img_qa_list = random.sample(img_qa_list, 30000)
    box_qa_list = random.sample(box_qa_list, 30000*cons_box_per_img)
    # 保存QA训练数据（与原逻辑一致）
    box_qa_output_path = os.path.join(output_json_dir, "Syn_JTEQAv2_box_anno.json")
    img_qa_output_path = os.path.join(output_json_dir, "Syn_JTEQAv2_img_anno.json")
    with open(box_qa_output_path, "w", encoding="utf-8") as f:
        json.dump(box_qa_list, f, ensure_ascii=False, indent=2)
    with open(img_qa_output_path, "w", encoding="utf-8") as f:
        json.dump(img_qa_list, f, ensure_ascii=False, indent=2)
    print("QA训练数据生成完成")
    # 新增数据统计模块
    total_images = len(raw_data_list)
    total_img_qa = len(img_qa_list)
    total_box_qa = len(box_qa_list)
    total_samples = total_img_qa + total_box_qa
    print("\n===== 数据统计结果 =====")
    print(f"处理图片数量：{total_images} 张")
    print(f"生成图像级QA样本数：{total_img_qa} 条")
    print(f"生成框级QA样本数：{total_box_qa} 条")
    print(f"总QA样本数：{total_samples} 条")

   





def pre_c(string):
    # 使用正则表达式替换所有空白字符，包括空格和换行符
    return re.sub(r'\s+', '', string)

def visualize_temp(image_path, box_coords, anno_text, converted_text, save_folder='./temp_visualization', image_id="unknown", box_id=0,hash_ratio=1):
    """
    临时可视化单个图像和边界框

    :param image_path: 图片文件路径
    :param box_coords: 框的坐标，格式为 [x_min, y_min, x_max, y_max]
    :param anno_text: 原始标注文本
    :param converted_text: 转换后的文本
    :param save_folder: 保存可视化结果的文件夹路径
    :param image_id: 图片的ID
    :param box_id: 框的ID
    """
    # 创建保存目录
    os.makedirs(save_folder, exist_ok=True)

    try:
        # 打开图像
        image = Image.open(image_path)
        img_width, img_height = image.size
        target_size = 1024
        width_ratio = target_size / img_width
        height_ratio = target_size / img_height
        # 调整图像大小以适配显示
        image = image.resize((target_size, target_size))
    except FileNotFoundError:
        print(f"未找到图片文件: {image_path}")
        return
    except Exception as e:
        print(f"打开图片失败: {e}")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')

    # 计算长宽比
    x_min, y_min, x_max, y_max = box_coords
    width = x_max - x_min
    height = y_max - y_min
    # 避免除以零的情况
    aspect_ratio = width / height if height > 0 else 0
    
    # 缩放框坐标以适配调整后的图像
    scaled_min_x = x_min * width_ratio
    scaled_min_y = y_min * height_ratio
    scaled_max_x = x_max * width_ratio
    scaled_max_y = y_max * height_ratio
    
    # 确保坐标在图像范围内
    scaled_min_x = max(0, min(scaled_min_x, target_size))
    scaled_min_y = max(0, min(scaled_min_y, target_size))
    scaled_max_x = max(0, min(scaled_max_x, target_size))
    scaled_max_y = max(0, min(scaled_max_y, target_size))
    
    min_x, min_y, max_x, max_y = map(int, [scaled_min_x, scaled_min_y, scaled_max_x, scaled_max_y])
    
    # 绘制框（使用绿色边框）
    rect = patches.Rectangle(
        (min_x, min_y), max_x - min_x, max_y - min_y,
        linewidth=2, edgecolor='#00FF00', facecolor='none', alpha=0.5
    )
    ax.add_patch(rect)

    # 添加文本信息，包含aspect ratio
    display_text = (f"Image ID: {image_id}\n" 
                   f"Box ID: {box_id}\n" 
                   f"Aspect Ratio: {aspect_ratio:.2f}\n" 
                   f"Hash Ratio: {hash_ratio:.2f}\n"
                   f"原始文本: {anno_text}\n" 
                   f"转换后文本: {converted_text}")
    
    # 在图像上方添加文本框，并使用指定的中文字体
    plt.figtext(0.5, 0.01, display_text, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8), 
                fontsize=10, 
                fontproperties=zhfont1)  # 使用指定的中文字体

    # 保存图像
    filename = os.path.join(save_folder, f'visualization_{image_id}_box{box_id}.png')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
def process_data(current_item,cons_box_per_img,ROOT):
    global BOX_IDX, IMG_IDX,bad_counter
    """
    处理单个图像的原始数据，生成图像级和框级QA样本
    :param current_item: 当前图像的原始数据（包含image_path/scaled_bboxs/ori_bboxs/annotations）
    :return: (img_qa_samples: 图像级样本列表, box_qa_samples: 框级样本列表)
    """
    # current_item = {
    #         "image_path": image_path,
    #         "scaled_bboxs": scaled_bboxs,
    #         "ori_bboxs": ori_bboxs,
    #         "annotations": annotations,
    #         "id" : project_id,
    #     }
    img_qa_samples = []
    box_qa_samples = []
    image_path = osp.join(ROOT,current_item['image_path'])
    oriid=current_item['id']
    constructed_img_answer = ""
    all_anno_texts = []  # 收集所有框的原始OCR文本（用于整体计算）
    boxes_for_sorting = []
    leave_img = True
    leave_box = True

    # ---------------------- 框级QA样本生成（每个框生成1个样本） ----------------------
    for idx, (ori_bbox, scaled_bbox, anno_text) in enumerate(zip(
        current_item["ori_bboxs"],  # 原始框坐标（与convert_training_textevalv2的ori_bbox对应）
        current_item["ori_bboxs"],  # 缩放后的框坐标（与question中的bbox_2d对应）
        current_item["annotations"]  # 该框的OCR文本（与识别结果对应）
    )):
    
        # converted_text= process_anno(anno_text)
        converted_text = anno_text
        y_min = scaled_bbox[1]  # scaled_bbox格式：[x_min, y_min, x_max, y_max]
        x_min = scaled_bbox[0]  
        boxes_for_sorting.append((y_min, x_min, converted_text))  #
        if leave_box:
        
            constructed_question = f'''
            This is a text-generated image. Please recognize all visible text in the local area "bbox_2d:{scaled_bbox}". 
            Marking rules:
            1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
            2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
            Output in the following JSON format:
            {{
            "recognized_text": "Text in "bbox_2d:{scaled_bbox}" (including structural error markers)"
            }}
            '''
            # 英文answer：对应question的字段，保持格式一致
            constructed_answer = {
                "recognized_text": f"{converted_text}"
            }
            constructed_answer_str = json.dumps(constructed_answer, ensure_ascii=False)
            constructed_answer_str = f"```json\n{constructed_answer_str}\n```"
            # constructed_img_answer += f'''识别区域 "bbox_2d:{scaled_bbox}"的文字内容为："{converted_text}"。\n'''
            all_anno_texts.append(anno_text)  # 收集原始OCR文本
            box_qa_conversation = [
                    {"from": "human", "value": f"<image>\n{constructed_question}"},
                    {"from": "gpt", "value": constructed_answer_str}#
                ]
            final_id = f'box_se_qa_{BOX_IDX}'
            BOX_IDX += 1
            box_qa_item = {
                "id": final_id,  # 基于图像路径和框索引生成唯一ID
                "image": image_path,
                "conversations": box_qa_conversation,
                "ori_bbox": ori_bbox , # 保留原始框坐标（与convert_training_textevalv2的ori_bbox字段一致）
            }
            if leave_box:
                box_qa_samples.append(box_qa_item)
    #随机抽样 box_qa_samples 保留 cons_box_per_img个 新shuffle 然后索引
    random.shuffle(box_qa_samples)
    # 2. 确定保留数量（取样本总数与cons_box_per_img的最小值）
    keep_count = min(len(box_qa_samples), cons_box_per_img)
    # 3. 截取前keep_count个样本
    box_qa_samples = box_qa_samples[:keep_count]
    # print(len(box_qa_samples))
    

    if leave_img:
       
        # ---------------------- 图像级QA样本生成 ----------------------
        # 图像级只有1个样本（整图分析）
        # 拼接所有框的原始OCR文本（用空格分隔）
        constructed_question = f'''
        This is a text-generated image. Please recognize all visible text in the entire image.
        Marking rules:
        1. Use <#> for structurally flawed (e.g., extra/missing strokes, distortion) unrecognizable Chinese characters or single English letters;
        2. Use <###> exclusively for structurally flawed unrecognizable single English words (not multi-word phrases, lines, or sentences).
        Output in the following JSON format:
        {{
        "recognized_text": "All text in the image (including structural error markers)"
        }}
        '''
        
        
        # 新增独立步骤一：按自然阅读顺序排序所有框文本（上到下、左到右）
        boxes_for_sorting.sort(key=lambda box: (box[0], box[1]))
        sorted_texts = [box[2] for box in boxes_for_sorting]  # 提取排序后的文本列表
        joined_texts = '\n'.join(sorted_texts)
        # 图像级英文answer：使用统一的recognized_text字段
        constructed_img_answer = {
            "recognized_text": f"{joined_texts}"
        }
        constructed_img_answer_str = json.dumps(constructed_img_answer, ensure_ascii=False)
        constructed_img_answer_str = f"```json\n{constructed_img_answer_str}\n```"
        # constructed_img_answer += f'''综上所述，该图片的文字质量得分为"quality_score":{img_quality_score}。'''
        img_qa_conversation = [
                    {"from": "human", "value": f"<image>\n{constructed_question}"},
                    {"from": "gpt", "value": constructed_img_answer_str}
                ]
        final_id = f'img_se_qa_{IMG_IDX}'
        IMG_IDX += 1
        img_qa_item = {
            "id": final_id,  # 基于图像路径生成唯一ID
            "image": image_path,
            "conversations": img_qa_conversation,
        }
        img_qa_samples.append(img_qa_item)
    else:
        img_qa_samples=[]

    return img_qa_samples, box_qa_samples
...

if __name__ == "__main__":
    
    RAW_DATA_PATH = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/synthtiger/syth_text_data.jsonl'# 需替换为实际数据路径
    OUTPUT_JSON_DIR = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/synthtiger/'# 需替换为实际数据路径
    cons_box_per_img = 5
    generate_qa_from_raw_data(RAW_DATA_PATH, OUTPUT_JSON_DIR,cons_box_per_img)
    print(f'bad_counter: {bad_counter}')
    