import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os.path as osp
import time
import argparse
import threading  # 新增：导入线程模块（用于超时控制）
import sys
from tqdm import tqdm
# 导入需要的库用于生成漂亮表格
from prettytable import PrettyTable

def normalized_edit_distance(s1, s2):
    """Calculate the normalized edit distance (NED) between two strings."""
    len_s1 = len(s1)
    len_s2 = len(s2)
    max_len = max(len_s1, len_s2)
    if max_len == 0:
        return 0.0
    # Calculate the edit distance
    dp = np.zeros((len_s1 + 1, len_s2 + 1))
    for i in range(len_s1 + 1):
        for j in range(len_s2 + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1,      # Deletion
                               dp[i][j-1] + 1,      # Insertion
                               dp[i-1][j-1] + cost) # Substitution
    # Normalize the edit distance
    ned = dp[len_s1][len_s2] / max_len
    return ned

def read_jsonl_file(file_path):
    """读取 JSONL 文件并返回其内容列表"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def normalized_edit_distance(s1, s2):
    """计算两个字符串的归一化编辑距离"""
    # 处理空字符串的情况
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 1.0

    # 计算编辑距离
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    # 归一化编辑距离
    max_len = max(len(s1), len(s2))
    return dp[len(s1)][len(s2)] / max_len

def run_evaluation(result_file, model_name):
    """加载OCR结果并计算指标，按data_source和语言分类统计"""
    # 定义data_source列表
    data_sources = [
        # "anno v2_box", 
        # "anno v2_image", 
        "anno v1_box", 
        "anno v1_image", 
        "anno advs_box", 
        "anno advs_image", 
        "sythetic_zaozi_box", 
        "sythetic_zaozi_image", 
        "sythetic_box", 
        "sythetic_image" 
    ]
    
    # 定义大类映射关系
    category_mapping = {
        # "anno v2_box": "anno_box",
        # "anno v2_image": "anno_image",
        "anno v1_box": "anno_box",
        "anno v1_image": "anno_image",
        "anno advs_box": "anno_box",
        "anno advs_image":  "anno_image",
        "sythetic_zaozi_box": "zaozi_box",
        "sythetic_zaozi_image": "zaozi_image",
        "sythetic_box": "render_box",
        "sythetic_image": "render_image"
    }
    
    # 初始化多层级指标字典结构
    # 结构: full_info[lang_mode][image_or_box][model_name][data_source][metric]
    full_info = {
        'chinese': {
            'box': {},
            'image': {}
        },
        'english': {
            'box': {},
            'image': {}
        }
    }
    
    # 为每种语言、image/box类型和数据源初始化指标计数器
    for lang_mode in ['chinese', 'english']:
        for img_box_type in ['box', 'image']:
            full_info[lang_mode][img_box_type][model_name] = {}
            
            # # 初始化所有数据源的指标
            # for ds in data_sources:
            #     full_info[lang_mode][img_box_type][model_name][ds] = {
            #         'global_hash_TP': 0,
            #         'global_hash_FP': 0,
            #         'global_hash_FN': 0,
            #         'global_hash_TN': 0,
            #         'total_correct_text': 0,
            #         'total_positive_pred': 0,
            #         'total_edit_distance': 0
            #     }
            
            # 初始化大类的指标
            for category in ['anno_box', 'anno_image', 'zaozi_box', 'zaozi_image', 'render_box', 'render_image',]:
                if category.endswith(img_box_type):
                    full_info[lang_mode][img_box_type][model_name][category] = {
                        'global_hash_TP': 0,
                        'global_hash_FP': 0,
                        'global_hash_FN': 0,
                        'global_hash_TN': 0,
                        'total_correct_text': 0,
                        'total_positive_pred': 0,
                        'total_edit_distance': 0
                    }
            
            # 初始化总体指标
            full_info[lang_mode][img_box_type][model_name]['total'] = {
                'global_hash_TP': 0,
                'global_hash_FP': 0,
                'global_hash_FN': 0,
                'global_hash_TN': 0,
                'total_correct_text': 0,
                'total_positive_pred': 0,
                'total_edit_distance': 0
            }
    
    # 处理结果文件
    if not osp.exists(result_file):
        print(f"Warning: File {result_file} not found, skipping...")
        return full_info
        
    print(f"Processing results from {result_file}")
    results = read_jsonl_file(result_file)
    
    # 定义语言判断函数
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'
        
    def contain_chinese(textline):
        return any(is_chinese(chr) for chr in textline)
        
    def is_english_text(text):
        """
        判断文本是否为纯英文（全英文返回True，含任何中文字符返回False）
        """
        # 检查文本中是否包含任何中文字符
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符Unicode范围
                return False  # 含中文 → 不是纯英文
        # 无中文字符 → 判定为英文
        return True
    
    def extract_text_info(text_list):
        print_mode=False
        clean_text_list = []  
        if print_mode:
            print(f'ori text list:{text_list}')
        for word in text_list:
            if '#' in word:
                if contain_chinese(word):
                    for ch in word:
                        if ch!='#' and is_chinese(ch) and len(ch)>0:
                            clean_text_list.append(ch)
                            if print_mode:
                                print(f'add {ch}')
                else:
                    continue
            elif contain_chinese(word):
                for ch in word:
                    if is_chinese(ch) and len(ch)>0:
                        clean_text_list.append(ch)
                        if print_mode:
                            print(f'add {ch}')
            else:#en
                clean_text_list.append(word)
                if print_mode:
                    print(f'add {word}')
        return clean_text_list
    
    # 遍历结果并统计
    for item in tqdm(results, desc=f'Calculating metrics for {osp.basename(result_file)}'):
        rec_text = item.get('pred_text', '')
        gt_rec_text = item.get('gt_text', '')
        data_source = item.get('data_source', 'unknown')
        idx = item.get('id','unknown')
        if 'box' not in idx and 'anno' in data_source and 'advs' not in data_source:
            a = 1 #debug
        # 确保data_source有效
        if data_source not in data_sources:
            print(f"Warning: Unknown data_source '{data_source}', skipping...")
            continue
        
        # 确定语言类型
        lang_type = 'english' if is_english_text(gt_rec_text) else 'chinese'
        
        # 确定image/box类型
        img_box_type = 'box' if 'box' in idx else 'image'
        # if img_box_type=='image':
        #    #debug
        #    a=1
        
        # 获取大类分类
        category = category_mapping[data_source]
        
        # 计算SAD指标
        pred_hash = rec_text.count('#')
        gt_hash = gt_rec_text.count('#')
        
        # 更新各个级别的统计数据
        def update_metrics(data_type):
            if pred_hash > 0:
                # if 0.5 * gt_hash <= pred_hash <= 2 * gt_hash:
                if 0.7 * gt_hash <= pred_hash <= 1/0.7 * gt_hash:
                    full_info[lang_type][img_box_type][model_name][data_type]['global_hash_TP'] += 1
                else:
                    full_info[lang_type][img_box_type][model_name][data_type]['global_hash_FP'] += 1
            else:
                if gt_hash == 0:
                    full_info[lang_type][img_box_type][model_name][data_type]['global_hash_TN'] += 1
                else:
                    full_info[lang_type][img_box_type][model_name][data_type]['global_hash_FN'] += 1
        
        # 更新原始数据源、大类和总汇总的统计
        # update_metrics(data_source)
        update_metrics(category)
        update_metrics('total')
        
        # 计算识别指标
        rec_text = re.sub(r'[<>]', '', rec_text)
        gt_rec_text = re.sub(r'[<>]', '', gt_rec_text)
        pred_text_list = rec_text.split(" ")
        gt_text_list = gt_rec_text.split(" ")
        clean_pred_text_list = extract_text_info(pred_text_list)
        clean_gt_text_list = extract_text_info(gt_text_list)
        
        # 定义更新识别指标的函数
        def update_recog_metrics(data_type, correct_text, positive_pred, edit_dist):
            full_info[lang_type][img_box_type][model_name][data_type]['total_correct_text'] += correct_text
            full_info[lang_type][img_box_type][model_name][data_type]['total_positive_pred'] += positive_pred
            full_info[lang_type][img_box_type][model_name][data_type]['total_edit_distance'] += edit_dist
            
       
        # 统计识别指标
        for correct_gt_text in clean_gt_text_list:
            current_correct = 1
            current_positive = 1 if correct_gt_text in clean_pred_text_list else 0
            current_edit = 0.0
            
            if correct_gt_text in clean_pred_text_list:
                current_edit = 0.0
            elif not contain_chinese(correct_gt_text):
                # 英文文本计算归一化编辑距离
                if clean_pred_text_list:
                    distances = [normalized_edit_distance(correct_gt_text, pred_text) for pred_text in clean_pred_text_list]
                    current_edit = min(distances)
                else:
                    current_edit = 1.0
            else:
                current_edit = 1.0
            
            # 更新各个级别的识别指标
            # update_recog_metrics(data_source, current_correct, current_positive, current_edit)
            update_recog_metrics(category, current_correct, current_positive, current_edit)
            update_recog_metrics('total', current_correct, current_positive, current_edit)
    
    # 计算各个分类的指标结果
    # 先计算每个数据源的指标
    for lang_mode in ['chinese', 'english']:
        for img_box_type in ['box', 'image']:
            for data_type in full_info[lang_mode][img_box_type][model_name]:
                metrics = full_info[lang_mode][img_box_type][model_name][data_type]
                
                # 计算 Precision
                precision = metrics['global_hash_TP'] / (metrics['global_hash_TP'] + metrics['global_hash_FP']) if (metrics['global_hash_TP'] + metrics['global_hash_FP']) > 0 else 0.0
                
                # 计算 Recall
                recall = metrics['global_hash_TP'] / (metrics['global_hash_TP'] + metrics['global_hash_FN']) if (metrics['global_hash_TP'] + metrics['global_hash_FN']) > 0 else 0.0
                
                # 计算 F1-score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # 计算识别 Recall 和 NED
                recog_recall = metrics['total_positive_pred'] / metrics['total_correct_text'] if metrics['total_correct_text'] > 0 else 0.0
                recog_ned = metrics['total_edit_distance'] / metrics['total_correct_text'] if metrics['total_correct_text'] > 0 else 0.0
                
                # 存储计算结果
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1_score'] = f1_score
                metrics['recog_recall'] = recog_recall
                metrics['recog_ned'] = recog_ned
    
    return full_info

def generate_summary_table(model_results, output_file, show_dataset_details=False, language_filter=None):
    """为所有模型结果生成漂亮的表格并写入文件
    
    Args:
        model_results: 模型结果字典
        output_file: 输出文件路径
        show_dataset_details: 是否显示数据集详情
        language_filter: 语言过滤，可以是 'chinese' 或 'english'，None 表示不过滤
    """
    # 决定生成哪种类型的表格
    if show_dataset_details:
        # 创建包含数据集详情的表格
        all_table = PrettyTable()
        # | | | five for image level | five for box level
        all_table.field_names = ["Model", "Data Type", "lang", "Precision_i", " Recall_i", " F1_i",  "Precision_b", " Recall_b", " F1_b", "recog recall_i", "ned_i",  "recog recall_b", "ned_b"]
        
        # 遍历所有模型结果
        for model_name, full_info in model_results.items():
            if full_info is not None:
                # 处理总体结果
                # 应用语言过滤
                languages = ['chinese'] if language_filter == 'chinese' else ['english'] if language_filter == 'english' else ['english', 'chinese']
                for lang in languages:
                    # 获取image和box级别的指标
                    img_metrics = calculate_level_metrics(full_info, model_name, lang, 'image')
                    box_metrics = calculate_level_metrics(full_info, model_name, lang, 'box')
                    
                    # 添加行
                    all_table.add_row([
                        model_name,
                        "Overall",
                        lang,
                        f"{img_metrics['precision']:.3f}",
                        f"{img_metrics['recall']:.3f}",
                        f"{img_metrics['f1_score']:.3f}",
                        f"{box_metrics['precision']:.3f}",
                        f"{box_metrics['recall']:.3f}",
                        f"{box_metrics['f1_score']:.3f}",
                        f"{img_metrics['recog_recall']:.3f}",
                        f"{img_metrics['recog_ned']:.3f}",
                        f"{box_metrics['recog_recall']:.3f}",
                        f"{box_metrics['recog_ned']:.3f}"
                    ])
                
                # 处理各个数据集类型的结果
                for data_type in ['anno', 'sythentic', 'zaozi']:
                    for lang in languages:
                        # 获取image和box级别的指标
                        img_metrics = calculate_dataset_level_metrics(full_info, model_name, lang, data_type, 'image')
                        box_metrics = calculate_dataset_level_metrics(full_info, model_name, lang, data_type, 'box')
                        
                        # 添加行
                        all_table.add_row([
                            model_name,
                            data_type,
                            lang,
                            f"{img_metrics['precision']:.3f}",
                            f"{img_metrics['recall']:.3f}",
                            f"{img_metrics['f1_score']:.3f}",
                            f"{box_metrics['precision']:.3f}",
                            f"{box_metrics['recall']:.3f}",
                            f"{box_metrics['f1_score']:.3f}",
                            f"{img_metrics['recog_recall']:.3f}",
                            f"{img_metrics['recog_ned']:.3f}",
                            f"{box_metrics['recog_recall']:.3f}",
                            f"{box_metrics['recog_ned']:.3f}"
                        ])
            else:
                # 对于无效的模型结果，添加空行
                for lang in languages:
                    all_table.add_row([model_name, "N/A", lang, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    else:
        # 创建标准表格，只显示总体和语言分类结果
        all_table = PrettyTable()
        all_table.field_names = ["Model", "lang", "Precision_i", " Recall_i", " F1_i", "Precision_b", " Recall_b", " F1_b", "recog recall_i", "ned_i", "recog recall_b", "ned_b"]
        
        # 遍历所有模型结果
        for model_name, full_info in model_results.items():
            if full_info is not None:
                # 应用语言过滤
                languages = ['chinese'] if language_filter == 'chinese' else ['english'] if language_filter == 'english' else ['english', 'chinese']
                # 为每种语言生成一行
                for lang in languages:
                    # 获取image和box级别的指标
                    img_metrics = calculate_level_metrics(full_info, model_name, lang, 'image')
                    box_metrics = calculate_level_metrics(full_info, model_name, lang, 'box')
                    
                    # 添加行
                    all_table.add_row([
                        model_name,
                        lang,
                        f"{img_metrics['precision']:.3f}",
                        f"{img_metrics['recall']:.3f}",
                        f"{img_metrics['f1_score']:.3f}",
                        f"{box_metrics['precision']:.3f}",
                        f"{box_metrics['recall']:.3f}",
                        f"{box_metrics['f1_score']:.3f}",
                        f"{img_metrics['recog_recall']:.3f}",
                        f"{img_metrics['recog_ned']:.3f}",
                        f"{box_metrics['recog_recall']:.3f}",
                        f"{box_metrics['recog_ned']:.3f}"
                    ])
            else:
                # 对于无效的模型结果，添加空行
                for lang in languages:
                    all_table.add_row([model_name, lang, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
    
    # 生成汇总内容
    summary_content = []
    summary_content.append("")
    if language_filter:
        summary_content.append(f"===== {language_filter} 指标汇总 ======")
    else:
        summary_content.append("===== 所有指标汇总 ======")
    summary_content.append(str(all_table))
    summary_content.append("")
    
    # 打印到控制台
    print("\n===== 汇总结果 =====")
    for line in summary_content:
        print(line)
    print("====================\n")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as summary_f:
        for line in summary_content:
            summary_f.write(f"{line}\n")
    
    print(f"汇总结果已保存到 {output_file}")


def calculate_level_metrics(full_info, model_name, language, img_box_type):
    """计算特定语言和级别(image/box)的指标"""
    # 初始化指标字典
    total_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'recog_recall': 0.0,
        'recog_ned': 0.0
    }
    
    # 检查是否存在该模型、语言和级别的数据
    if model_name in full_info[language][img_box_type] and 'total' in full_info[language][img_box_type][model_name]:
        metrics = full_info[language][img_box_type][model_name]['total']
        
        # 直接提取已计算好的指标
        total_metrics['precision'] = metrics.get('precision', 0.0)
        total_metrics['recall'] = metrics.get('recall', 0.0)
        total_metrics['f1_score'] = metrics.get('f1_score', 0.0)
        total_metrics['recog_recall'] = metrics.get('recog_recall', 0.0)
        total_metrics['recog_ned'] = metrics.get('recog_ned', 0.0)
    
    return total_metrics


def calculate_dataset_level_metrics(full_info, model_name, language, dataset_type, img_box_type):
    """计算特定数据集类型、语言和级别的指标"""
    # 初始化指标字典
    total_metrics = {
        'global_hash_TP': 0,
        'global_hash_FP': 0,
        'global_hash_FN': 0,
        'global_hash_TN': 0,
        'total_correct_text': 0,
        'total_positive_pred': 0,
        'total_edit_distance': 0
    }
    
    # 根据数据集类型确定要聚合的类别
    if dataset_type == 'anno':
        categories = ['anno_box', 'anno_image','advs_box', 'advs_image']
    elif dataset_type == 'zaozi':
        categories = ['zaozi_box', 'zaozi_image']
    elif dataset_type == 'sythentic':
        categories = ['render_box', 'render_image']
    else:
        categories = []
    
    # 累加对应类别的指标
    for category in categories:
        if category.endswith(img_box_type) and model_name in full_info[language][img_box_type] and category in full_info[language][img_box_type][model_name]:
            metrics = full_info[language][img_box_type][model_name][category]
            
            total_metrics['global_hash_TP'] += metrics.get('global_hash_TP', 0)
            total_metrics['global_hash_FP'] += metrics.get('global_hash_FP', 0)
            total_metrics['global_hash_FN'] += metrics.get('global_hash_FN', 0)
            total_metrics['global_hash_TN'] += metrics.get('global_hash_TN', 0)
            total_metrics['total_correct_text'] += metrics.get('total_correct_text', 0)
            total_metrics['total_positive_pred'] += metrics.get('total_positive_pred', 0)
            total_metrics['total_edit_distance'] += metrics.get('total_edit_distance', 0.0)
    
    # 计算各项指标
    precision = total_metrics['global_hash_TP'] / (total_metrics['global_hash_TP'] + total_metrics['global_hash_FP']) if (total_metrics['global_hash_TP'] + total_metrics['global_hash_FP']) > 0 else 0.0
    recall = total_metrics['global_hash_TP'] / (total_metrics['global_hash_TP'] + total_metrics['global_hash_FN']) if (total_metrics['global_hash_TP'] + total_metrics['global_hash_FN']) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    recog_recall = total_metrics['total_positive_pred'] / total_metrics['total_correct_text'] if total_metrics['total_correct_text'] > 0 else 0.0
    recog_ned = total_metrics['total_edit_distance'] / total_metrics['total_correct_text'] if total_metrics['total_correct_text'] > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'recog_recall': recog_recall,
        'recog_ned': recog_ned
    }

def calculate_overall_metrics(full_info, model_name):
    """计算模型的总体指标（用于主程序）"""
    total_metrics = {
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'recog_recall': 0,
        'recog_ned': 0,
        'global_hash_TP': 0,
        'global_hash_FP': 0,
        'global_hash_FN': 0,
        'global_hash_TN': 0,
        'total_correct_text': 0,
        'total_positive_pred': 0,
        'total_edit_distance': 0
    }
    
    # 累加所有语言和类型的指标
    for lang_mode in ['chinese', 'english']:
        for img_box_type in ['box', 'image']:
            if model_name in full_info[lang_mode][img_box_type] and 'total' in full_info[lang_mode][img_box_type][model_name]:
                model_total = full_info[lang_mode][img_box_type][model_name]['total']
                
                total_metrics['global_hash_TP'] += model_total['global_hash_TP']
                total_metrics['global_hash_FP'] += model_total['global_hash_FP']
                total_metrics['global_hash_FN'] += model_total['global_hash_FN']
                total_metrics['global_hash_TN'] += model_total['global_hash_TN']
                total_metrics['total_correct_text'] += model_total['total_correct_text']
                total_metrics['total_positive_pred'] += model_total['total_positive_pred']
                total_metrics['total_edit_distance'] += model_total['total_edit_distance']
    
    # 重新计算总体指标
    # 计算 Precision
    total_metrics['precision'] = total_metrics['global_hash_TP'] / (total_metrics['global_hash_TP'] + total_metrics['global_hash_FP']) if (total_metrics['global_hash_TP'] + total_metrics['global_hash_FP']) > 0 else 0.0
    
    # 计算 Recall
    total_metrics['recall'] = total_metrics['global_hash_TP'] / (total_metrics['global_hash_TP'] + total_metrics['global_hash_FN']) if (total_metrics['global_hash_TP'] + total_metrics['global_hash_FN']) > 0 else 0.0
    
    # 计算 F1-score
    total_metrics['f1_score'] = 2 * (total_metrics['precision'] * total_metrics['recall']) / (total_metrics['precision'] + total_metrics['recall']) if (total_metrics['precision'] + total_metrics['recall']) > 0 else 0.0
    
    # 计算识别 Recall 和 NED
    total_metrics['recog_recall'] = total_metrics['total_positive_pred'] / total_metrics['total_correct_text'] if total_metrics['total_correct_text'] > 0 else 0.0
    total_metrics['recog_ned'] = total_metrics['total_edit_distance'] / total_metrics['total_correct_text'] if total_metrics['total_correct_text'] > 0 else 0.0
    
    return total_metrics

if __name__ == '__main__':
    # 添加命令行参数解析
    #TODO 1.10 改成支持hf data格式 避免advs 进行分类
    parser = argparse.ArgumentParser(description='Calculate metrics for OCR results')
    parser.add_argument('--language', choices=['chinese', 'english', 'all'], default='all', 
                        help='Filter results by language (chinese, english, or all)')
    args = parser.parse_args()
    
    # 根据命令行参数设置 language_filter
    language_filter = args.language if args.language != 'all' else None
    

    # 模型名称和对应的结果文件路径列表
    model_name_list = [
        # ！the name you want to display in the final result
        'textpecker-it'
    ]
    
    # 注意：这里需要根据实际情况修改为正确的文件路径
    answer_path_list = [
        #！your final inference result json path
        '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/train/ms-swift/inference-output/eval_results_it_converted.jsonl'
        
    ]
    
    #denote the output dir you want to save the table results
    eval_output_dir = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/train/ms-swift/inference-output/' #replace with you own
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # 总的输出文件
    summary_output_file = osp.join(eval_output_dir, 'metrics_summary.txt')
    
    # 收集所有模型的详细结果
    detailed_model_results = {}
    
    # 收集所有模型的总体结果（用于兼容原有的表格生成）
    overall_model_results = {}
    
    # 确保model_name_list和answer_path_list长度匹配
    if len(model_name_list) != len(answer_path_list):
        print("Error: model_name_list and answer_path_list must have the same length!")
        sys.exit(1)
    
    # 遍历每个模型，计算指标
    for i in range(len(model_name_list)):
        model_name = model_name_list[i]
        result_file = answer_path_list[i]
        print(f"\nProcessing model: {model_name}")
        
        # 计算详细指标
        detailed_metrics = run_evaluation(result_file, model_name)
        detailed_model_results[model_name] = detailed_metrics
    
    
    # 生成详细汇总表格
    # 如果只有一个模型，默认显示数据集详情
    show_dataset_details = len(model_name_list) == 1
    show_dataset_details = False
    # 传入 language_filter 参数
    generate_summary_table(detailed_model_results, summary_output_file, show_dataset_details, language_filter)
  