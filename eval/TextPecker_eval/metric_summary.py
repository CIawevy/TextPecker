#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict
import numpy as np
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize evaluation metrics from JSONL files')
    parser.add_argument('--result_dir', type=str, 
                        default='eval/TextPecker_eval/results',
                        help='Directory containing evaluation results')
    parser.add_argument('--output_file', type=str, 
                        default='metric_summary.txt',
                        help='Output file to save the summary')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=None,
                        help='List of model names to include in the summary. If not provided, all models will be included.')
    return parser.parse_args()


def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data


def calculate_metrics(data):
    """Calculate average metrics from the data"""
    if not data:
        return 0.0, 0.0, 0.0
    
    # sem_scores = []
    qua_scores = []
    ocr_scores = []
    gned_scores = []  # 新增：存储 pecker_gned 分数
    
    for item in data:
        # 处理JSON中的实际字段名
        if isinstance(item, dict):
            # 提取pecker_sem、pecker_qua和ocr_score字段
            # if 'pecker_sem' in item and item['pecker_sem'] is not None:
            #     sem_scores.append(item['pecker_sem'])
            
            if 'pecker_qua' in item and item['pecker_qua'] is not None:
                qua_scores.append(item['pecker_qua'])
            
            if 'ocr_score' in item and item['ocr_score'] is not None:
                ocr_scores.append(item['ocr_score'])
            
            # 新增：提取 pecker_gned 字段
            if 'pecker_gned' in item and item['pecker_gned'] is not None:
                gned_scores.append(item['pecker_gned'])
    
    # 计算平均值，如果没有数据则为0
    # avg_sem = np.mean(sem_scores) if sem_scores else 0.0
    avg_qua = np.mean(qua_scores) if qua_scores else 0.0
    avg_ocr = np.mean(ocr_scores) if ocr_scores else 0.0
    avg_gned = np.mean(gned_scores) if gned_scores else 0.0  # 新增：计算 pecker_gned 平均值
    
    return avg_qua, avg_ocr, avg_gned


def main():
    args = parse_args()
    result_dir = args.result_dir
    output_file = args.output_file
    models_to_include = args.models
    
    # 创建全局结果字典
    all_results = defaultdict(dict)
    
    # 遍历所有数据集类型 (cvtg, longtext, oneig)
    for dataset_type in os.listdir(result_dir):
        dataset_path = os.path.join(result_dir, dataset_type)
        
        # 检查是否是目录
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"Processing dataset: {dataset_type}")
        
        # 遍历每个数据集下的所有模型
        for model_name in os.listdir(dataset_path):
            # 如果指定了模型列表，并且当前模型不在列表中，则跳过
            if models_to_include is not None and model_name not in models_to_include:
                continue
                
            model_path = os.path.join(dataset_path, model_name)
            
            # 检查是否是目录
            if not os.path.isdir(model_path):
                continue
            
            # 查找eval_results.jsonl文件
            jsonl_path = os.path.join(model_path, 'eval_results.jsonl')
            
            if os.path.exists(jsonl_path):
                # 加载数据并计算指标
                data = load_jsonl(jsonl_path)
                avg_qua, avg_ocr, avg_gned = calculate_metrics(data)  # 新增：获取 pecker_gned 平均值
                if 'cvtg' in dataset_path or 'lex' in dataset_path or 'atlas' in dataset_path or 'tiif' in dataset_path:
                    model_name+='_EN'
                # 存储结果
                all_results[dataset_type][model_name] = {
                    # 'sem': avg_sem,
                    'qua': avg_qua,
                    # 'ocr': avg_ocr,
                    'gned': avg_gned,  # 新增：存储 pecker_gned 结果
                    'count': len(data)
                }
                
                print(f"  Model: {model_name}, Samples: {len(data)}, Qua: {avg_qua:.3f}, Gned: {avg_gned:.3f}")  # 修改为.3f
            else:
                print(f"  Model: {model_name}, No eval_results.jsonl found")
    
    # 生成美观的表格输出并保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 定义数据集顺序
        datasets = ['oneig', 'longtext', 'cvtg', 'gentexteval','lex','atlas','tiif']
        # 构建表格列名：Model + 每个数据集的Qua、Sem列
        field_names = ['Model']
        for ds in datasets:
            field_names.append(f"{ds}_Qua")
            field_names.append(f"{ds}_Sem")
        
        # 初始化PrettyTable
        table = PrettyTable(field_names)
        table.align = 'l'  # 模型列左对齐
        for col in field_names[1:]:  # 数值列右对齐
            table.align[col] = 'r'
        
        # 收集所有模型名称（去重并排序）
        all_models = set()
        for ds in datasets:
            if ds in all_results:
                all_models.update(all_results[ds].keys())
        all_models = sorted(all_models)
        
        # 为每个模型生成一行数据
        for model_name in all_models:
            row = [model_name]
            for ds in datasets:
                if ds in all_results and model_name in all_results[ds]:
                    metrics = all_results[ds][model_name]
                    qua = f"{metrics.get('qua', 0):.3f}"  # 取qua指标，保留3位小数
                    sem = f"{metrics.get('gned', 0):.3f}"  # 取sem指标，保留3位小数
                else:
                    qua = "NA"  # 无数据时填充NA
                    sem = "NA"
                row.append(qua)
                row.append(sem)
            table.add_row(row)
        
        # 打印并写入整合后的表格
        table_str = str(table)
        print(table_str)
        f.write(table_str + '\n\n')
        
        # 计算各数据集Qua、Sem的整体平均值
        overall_avg = []
        for i, ds in enumerate(datasets):
            qua_col = f"{ds}_Qua"
            sem_col = f"{ds}_Sem"
            qua_values = []
            sem_values = []
            for row in table._rows:
                qua_val = row[2*i + 1]
                sem_val = row[2*i + 2]
                if qua_val != "NA":
                    qua_values.append(float(qua_val))
                if sem_val != "NA":
                    sem_values.append(float(sem_val))
            avg_qua = f"{np.mean(qua_values):.3f}" if qua_values else "NA"
            avg_sem = f"{np.mean(sem_values):.3f}" if sem_values else "NA"
            overall_avg.append(f"{ds}: Qua={avg_qua}, Sem={avg_sem}")
        
        # 写入总体平均值
        overall_header = "===== OVERALL AVERAGE RESULTS ====="
        print(overall_header)
        print("\n".join(overall_avg))
        f.write(overall_header + '\n')
        f.write("\n".join(overall_avg) + '\n')

    print(f"\nSummary saved to {output_file}")
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     # 为每个数据集生成单独的表格
    #     for dataset_type in sorted(all_results.keys()):
    #         header = f"\n===== {dataset_type.upper()} Dataset Results ====="
    #         print(header)
    #         f.write(header + '\n')
            
    #         # 创建表格
    #         table = PrettyTable()
    #         table.field_names = ['Model', 'Samples', 'Qua Score', 'Gned Score']  # 新增：添加 Gned Score 列
            
    #         # 添加数据到表格
    #         for model_name in sorted(all_results[dataset_type].keys()):
    #             metrics = all_results[dataset_type][model_name]
    #             table.add_row([
    #                 model_name,
    #                 metrics['count'],
    #                 # f"{metrics['sem']:.3f}",
    #                 f"{metrics['qua']:.3f}",
    #                 # f"{metrics['ocr']:.3f}",
    #                 f"{metrics['gned']:.3f}"  # 修改为.3f
    #             ])
            
    #         # 设置表格样式
    #         table.align = 'l'  # 左对齐
    #         table.align['Samples'] = 'r'
    #         # table.align['Sem Score'] = 'r'
    #         # table.align['Qua Score'] = 'r'
    #         table.align['OCR Score'] = 'r'
    #         table.align['Gned Score'] = 'r'  # 新增：设置 Gned Score 列右对齐
            
    #         # 打印表格并写入文件
    #         table_str = str(table)
    #         print(table_str)
    #         f.write(table_str + '\n\n')
            
    #         # 计算并打印该数据集的平均得分
    #         dataset_metrics = []
    #         for metrics in all_results[dataset_type].values():
    #             dataset_metrics.append([metrics['qua'], metrics['gned']])  # 新增：包含 Gned 指标
            
    #         if dataset_metrics:
    #             avg_dataset_metrics = np.mean(dataset_metrics, axis=0)
    #             # avg_line = f"Average for {dataset_type}:  Qua={avg_dataset_metrics[0]:.3f}, Gned={avg_dataset_metrics[1]:.3f}"  # 修改为.3f
    #             # print(avg_line)
    #             # f.write(avg_line + '\n\n')
        
    #     # 生成总体汇总
    #     all_metrics = []
    #     for dataset_type in all_results:
    #         for metrics in all_results[dataset_type].values():
    #             all_metrics.append([metrics['qua'], metrics['gned']])  # 新增：包含 Gned 指标
        
    #     if all_metrics:
    #         overall_avg = np.mean(all_metrics, axis=0)
    #         overall_header = "\n===== OVERALL RESULTS ====="
    #         overall_line = f"Overall Average:  Qua={overall_avg[0]:.3f}, Gned={overall_avg[1]:.3f}"  # 修改为.3f
            
    #         print(overall_header)
    #         print(overall_line)
    #         f.write(overall_header + '\n')
    #         f.write(overall_line + '\n')
    
    # print(f"\nSummary saved to {output_file}")


if __name__ == '__main__':
    main()