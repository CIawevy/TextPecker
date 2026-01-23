import json
import os

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
            lines = list(f)
            if not lines:
                print(f"警告：文件 {file_path} 为空")
                return json_objects
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
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

# 测试代码
data_path = "/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextPecker/eval/TIIF-Bench/text_prompts.jsonl"

# 检查文件是否存在
if not os.path.exists(data_path):
    print(f"错误：文件不存在: {data_path}")
    print(f"当前工作目录: {os.getcwd()}")
else:
    print(f"文件存在，大小: {os.path.getsize(data_path)} 字节")
    
    try:
        # 加载JSONL文件
        data = load_jsonl(data_path)
        print(f"成功加载了 {len(data)} 个样本")
        
        # 打印前3个样本的简要信息
        if data:
            print("\n前3个样本的内容：")
            for i, item in enumerate(data[:3], 1):
                print(f"\n样本 {i}:")
                print(f"类型: {item.get('type', '未知')}")
                print(f"简短描述: {item.get('short_description', '未知')[:50]}...")
                print(f"长描述长度: {len(item.get('long_description', ''))} 字符")
                print(f"包含的键: {list(item.keys())}")
    except Exception as e:
        print(f"加载文件时出错: {e}")