import json
import os
import random
random.seed(42)
from tqdm import tqdm
import pandas as pd
import json
import os
from PIL import Image  # 新增图像操作库
import math  # 新增数学计算库
import re  # 新增正则表达式库
import numpy as np
from tqdm import tqdm
import random
import string
from typing import Tuple , List, Dict
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import sys
sys.path.append('/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger') #replace with your own
font_path = 'resources/Arial_Unicode.ttf'
font_prop = FontProperties(fname=font_path)

# 设置 matplotlib 使用该字体
plt.rcParams['font.family'] = font_prop.get_name()
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
def load_json_data(file_path):
    """加载JSON数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def rename_ids(data, prefix):
    """
    按照指定前缀重新编号数据的id
    :param data: 数据列表
    :param prefix: 编号前缀，如 'train' 或 'eval'
    :return: 重新编号后的数据列表
    """
    for index, item in enumerate(data):
        item["id"] = f"{prefix}_{index}"
    return data

def save_json_data(data, file_path):
    """保存JSON数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def split_eval_and_train(data, eval_size=400):
    """
    拆分验证集和训练集
    :param data: 原始数据列表
    :param eval_size: 验证集大小
    :return: (train_data, eval_data)
    """
    random.shuffle(data)
    return data[eval_size:], data[:eval_size]

def merge_and_shuffle_data(new_train_data, existing_train_path):
    """
    合并新生成训练数据与现有训练数据并打乱
    :param new_train_data: 新生成的训练数据
    :param existing_train_path: 现有训练数据路径
    :return: 合并并打乱后的训练数据
    """
    # 加载现有训练数据
    if os.path.exists(existing_train_path):
        existing_train_data = load_json_data(existing_train_path)
        print(f"加载现有训练数据: {len(existing_train_data)} 条")
        
        # 去重（基于id）
        id_set = set()
        unique_data = []
        for item in existing_train_data + new_train_data:
            if item["id"] not in id_set:
                id_set.add(item["id"])
                unique_data.append(item)
        print(f"合并后去重数据: {len(unique_data)} 条")
    else:
        unique_data = new_train_data
        print("未找到现有训练数据，仅使用新生成数据")
    
    # 最终打乱
    random.shuffle(unique_data)
    return unique_data

def main():
    # ======================== 配置参数 ========================
    # 新生成的box和img数据路径（请替换为实际路径）
    NEW_BOX_DATA_PATH = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger/zaozi_JTEQAv2_box_anno.json"  # 需替换为实际box数据路径
    NEW_IMG_DATA_PATH = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger/zaozi_JTEQAv2_img_anno.json"  # 需替换为实际img数据路径
  
    # 输出路径
    OUTPUT_DIR = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/data/synthtiger"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 验证集大小（各类型）
    # ==========================================================

    # 1. 加载新生成的box和img数据
    print("加载新生成数据...")
    new_box_data = load_json_data(NEW_BOX_DATA_PATH)
    # new_box_data_sem = load_json_data(NEW_BOX_DATA_PATH)
    new_img_data = load_json_data(NEW_IMG_DATA_PATH)

    
    

    # existing_sem_box_eval_ids = set([item['id'] for item in existing_sem_box_eval])
    # print(f'exisitinf id nums:{len(existing_sem_box_eval_ids)}')
    
    # 从新数据中筛选出不在验证集中的数据作为训练数据
    train_box_anno,eval_box_anno = split_eval_and_train(new_box_data,eval_size=50)
    train_img_anno,eval_img_anno =  split_eval_and_train(new_img_data,eval_size=50)
    
    
    
    # 重命名验证集ID
    eval_img_anno = rename_ids(eval_img_anno, "syn_annoeval_img")
    eval_box_anno = rename_ids(eval_box_anno, "syn_annoeval_box")
    train_img_anno = rename_ids(train_img_anno, "syn_annotrain_img")
    train_box_anno = rename_ids(train_box_anno, "syn_annotrain_box")
    # train_box_sem = rename_ids(train_box_sem, "syn_annotrain_box_sem")
    

    # 3. 混合新生成的训练数据
    # unique_data = train_box_anno + train_img_anno + train_box_sem
    unique_data = train_img_anno + train_box_anno
    
    
    random.shuffle(unique_data)

    
    # 5. 保存结果
    print("保存结果文件...")
    # 保存训练集
    # train_path = os.path.join(OUTPUT_DIR, f"syn_tiger_textqa_merge_{int(len(unique_data)/1000)}k.json")
    # save_json_data(
    #     unique_data,
    #     train_path
    # )
    save_json_data(train_img_anno, os.path.join(OUTPUT_DIR, "tJTEQAv2_img_anno_zaozi.json"))
    save_json_data(train_box_anno, os.path.join(OUTPUT_DIR, "tJTEQAv2_box_anno_zaozi.json"))
    # 保存验证集
    save_json_data(eval_img_anno, os.path.join(OUTPUT_DIR, "eJTEQAv2_img_anno_zaozi.json"))
    save_json_data(eval_box_anno, os.path.join(OUTPUT_DIR, "eJTEQAv2_box_anno_zaozi.json"))
    # save_json_data(eval_box_sem, os.path.join(OUTPUT_DIR, "eJTEQAv2_box_semantic.json"))



    # 打印统计信息
    print("\n===== 数据处理完成 =====")
    # print(f"最终训练集大小: {len(unique_data)} 条")
    # print(f"img验证集大小: {len(eval_img)} 条")
    # print(f"训练集保存至: {train_path}")
    # print(f"img验证集保存至: {os.path.join(OUTPUT_DIR, 'eJTEQAv2_img_anno.json')}")




if __name__ == "__main__":
    main()
    #注意先进入到data目录下image.png