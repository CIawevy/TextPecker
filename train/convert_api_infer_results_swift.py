import os
import json
import sys
from tqdm import tqdm
import re


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.append(_SCRIPT_DIR)

from parse_utils import get_score

def read_json_or_jsonl(file_path):
    """根据文件扩展名读取JSON或JSONL文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                return [json.loads(line) for line in f]
            else:  # 假设是JSON文件
                return json.load(f)
    except Exception as e:
        print(f"读取文件出错: {e}")
        return []

def extract_pred_text(response):
    """从response中提取预测的文本"""
    try:
        # 使用get_score函数获取预测识别文本
        _, _, pred_rec_text, _ = get_score('RFI', response, None)
        return pred_rec_text
    except Exception as e:
        print(f"提取预测文本时出错: {e}")
        return response

def extract_gt_from_infer(infer_item):
    """从推理结果项中提取GT文本"""
    try:
        # 检查是否有labels字段
        if 'labels' in infer_item:
            label = infer_item['labels']
            _, _, gt_text, _ = get_score('RFI', label, None)
            return gt_text
        
        # 如果都失败了，返回空字符串
        return ""
    except Exception as e:
        print(f"从推理结果提取GT文本时出错: {e}")
        return ""

def process_file(infer_file):
    """处理推理结果文件"""
    print(f"开始处理推理结果文件: '{infer_file}'")

    infer_data = read_json_or_jsonl(infer_file)

    if not infer_data:
        print("错误: 未能读取到有效推理结果数据")
        return

    converted_items = []
    processed_count = 0

    for infer_item in tqdm(infer_data, desc=f"处理推理结果项"):
        try:
            # 从推理结果中提取所有必要信息
            item_id = infer_item.get("id", "")
            data_source = infer_item.get("data_source", "")
            image_class = infer_item.get("class", "")
            
            # 提取预测文本
            response = infer_item.get("response", "")
            pred_text = extract_pred_text(response)
            
            # 提取真实文本
            gt_text = extract_gt_from_infer(infer_item)

            converted_item = {
                "image": image_class,  
                "id": item_id,
                "data_source": data_source,
                "pred_text": pred_text,
                "gt_text": gt_text,
            }
            converted_items.append(converted_item)
            processed_count += 1

        except Exception as e:
            print(f"处理推理结果项 {infer_item.get('id', 'unknown')} 时出错: {e}")

    # 构建输出文件路径
    dir_path = os.path.dirname(infer_file)
    base_name = os.path.basename(infer_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(dir_path, f"{name_without_ext}_converted.jsonl")
    
    # 保存转换后的结果
    print(f"\n保存转换后的结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 输出统计信息
    print(f"\n处理统计:")
    print(f"总推理结果项数: {len(infer_data)}")
    print(f"成功处理的项数: {processed_count}")
    print(f"处理失败的项数: {len(infer_data) - processed_count}")
    print(f"处理成功率: {processed_count/len(infer_data)*100:.2f}%")
    print(f"文件处理完成: {infer_file}")

def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python convert_api_infer_results_swift.py <推理结果文件路径>")
        print("示例: python convert_api_infer_results_swift.py /path/to/infer_results.jsonl")
        return
    
    # 获取推理结果文件路径
    infer_files = sys.argv[1:]
    
    # 逐个处理文件
    for infer_file in infer_files:
        if os.path.exists(infer_file):
            process_file(infer_file)
            print("=" * 80)
        else:
            print(f"警告: 文件未找到 - {infer_file}")
    
    print("所有文件处理完成。")

if __name__ == "__main__":
    main()