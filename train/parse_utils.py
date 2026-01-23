import re
import json
from typing import List

def save_jsonl_file(file_path, data):
    """保存数据到 JSONL 文件（每行一个JSON对象）"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
def read_json_file(file_path):
    """读取 JSON 文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def save_json_file(file_path, data):
    """保存数据到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, indent=4, ensure_ascii=False, fp=f)

def parse_quoted_text(prompt):
    """
    提取引号包裹的文本内容，支持中英文多种引号类型：
    - 中文双引号：“...””
    - 中文单引号：‘...’
    - 英文双引号："..."
    - 英文单引号：'...'
    使用贪心算法从左到右匹配引号对，提取所有顶层引号内容
    """
    # 定义引号映射表：左引号 -> 右引号
    quote_map = {
        '“': '”',   # 中文双引号
        '‘': '’',   # 中文单引号
        '"': '"',   # 英文双引号
        "'": "'",    # 英文单引号
        '《': '》'   # 书名号
    }
    
    results = []
    in_quote = False  # 引号内状态标记
    current_closing = None  # 当前等待匹配的右引号
    start_pos = 0  # 引号内容起始位置
    
    for i, char in enumerate(prompt):
        if not in_quote:
            # 遇到左引号时进入引号内状态
            if char in quote_map:
                in_quote = True
                current_closing = quote_map[char]
                start_pos = i + 1  # 内容从左引号后开始
        else:
            # 遇到对应右引号时提取内容
            if char == current_closing:
                # 提取引号间内容并去除首尾空白
                content = prompt[start_pos:i].strip()
                if content:  # 忽略空内容
                    results.append(content)
                # 重置引号状态
                in_quote = False
                current_closing = None
    
    return results

def parse_text_atlas(prompt):
    """
    改进版OCR文本提取函数，解决以下问题：
    1. 支持单引号 ('...'), 双引号 ("..."), 以及连续两个单引号 (''...'')
    2. 递归提取嵌套引号内的文本
    3. 正确处理转义字符（如 \", \'）
    4. 排除所有格形式（如 xxx's）
    5. 避免重复提取
    6. 跳过 'the text :' 前缀
    """
    import re

    def recursive_extract(text):
        results = []
        # 定义不同引号类型
        quote_types = ["''", '""', "'", '"']
        remaining_text = text
        for quote in quote_types:
            # 动态生成正则表达式，排除所有格形式并跳过 'the text :' 前缀
            pattern = (
                r'(?<!\w)'                     # 确保左侧不是字母（排除所有格）
                r'(?:the\s+text\s*:\s*)?'      # 可选的 'the text :' 前缀
                r'(?<!\\)(?:\\\\)*'            # 允许转义字符（如 \", \'）
                + re.escape(quote) + 
                r'(.*?)'                       # 非贪婪匹配内容
                r'(?<!\\)(?:\\\\)*'            # 允许转义字符
                + re.escape(quote) +
                r'(?!\w)'                      # 确保右侧不是字母（排除所有格）
            )
            while True:
                match = re.search(pattern, remaining_text, re.DOTALL)
                if not match:
                    break
                start, end = match.span()
                content = match.group(1)
                # 递归提取嵌套内容
                nested_results = recursive_extract(content)
                if nested_results:
                    results.extend(nested_results)
                else:
                    results.append(content.strip())
                # 移除已匹配的部分
                remaining_text = remaining_text[:start] + remaining_text[end:]
        return results

    all_extracted = recursive_extract(prompt)
    final_results = []
    for text in all_extracted:
        # 去重并保留最长有效文本
        if text and text not in final_results:
            final_results.append(text)
    return final_results


def parse_prompt(prompt,mode='cn'):
    if prompt is None:
        return None
    if mode=='cn':
        ground_truth = parse_quoted_text(prompt)
        ground_truth = ''.join(ground_truth)
    elif mode=='en':
        ground_truth = parse_text_atlas(prompt)
        ground_truth = ' '.join(ground_truth)
    else:
        raise ValueError(f'Unknown mode {mode}')
    return ground_truth



def get_score(task_name,string,ref_target=None):

    #todo:大幅度修改
    #1.parse json
    #2.extract corrext
    #3.extract ref
    #4. get score
    def process_raw_text(raw_text):
        if not raw_text:
            return ""
        
        # 第一步：统一处理原始文本格式（列表转字符串）
        if isinstance(raw_text, List):
            pro_text = ' '.join(raw_text)
        elif isinstance(raw_text, str):
            pro_text = raw_text
        else:
            return ""  # 非字符串/列表类型直接返回空
        
        # 第二步：还原 JSON 转义字符（关键！处理 \\", \\', \\n 等）
        pro_text = pro_text.replace('\\\\', '\\')  # 先还原双反斜杠为单反斜杠
        pro_text = pro_text.replace('\\n', '\n')   # 还原换行符
        pro_text = pro_text.replace('\\"', '"')    # 还原转义双引号
        pro_text = pro_text.replace("\\'", "'")    # 还原转义单引号
        
        # 第三步：重新统计引号数量（含转义后还原的引号）
        quote_chars = ["\"", "“", "‘", "'", "〝"]
        quote_count = sum(pro_text.count(char) for char in quote_chars)
        
        if quote_count >= 2:
            # 引号匹配模式：优化对混合引号和转义的处理
            result = []
            in_quote = False
            current_quote = None  # 记录当前打开的引号类型
            close_quote_mapping = {"\"": "\"", "“": "”", "‘": "’", "'": "'", "〝": "〞"}
            
            for char in pro_text:
                if char in close_quote_mapping.keys():
                    # 遇到左引号：如果不在引号内，标记为当前引号
                    if not in_quote:
                        in_quote = True
                        current_quote = char
                    # 遇到右引号：如果与当前引号匹配，关闭引号
                    elif char == close_quote_mapping.get(current_quote):
                        in_quote = False
                        current_quote = None
                    # 不匹配的引号：视为普通字符加入结果
                    else:
                        result.append(char)
                elif char in ["，", "。", "！", "？", ",", " "] and not in_quote:
                    # 非引号内的标点和空格统一转为空格
                    result.append(' ')
                else:
                    # 其他字符直接加入（包括引号内的标点）
                    result.append(char)
            
            # 合并结果并去除连续空格
            pro_text = ' '.join(''.join(result).split())
        else:
            # 无引号或引号不足时，直接处理标点
            pro_text = re.sub(r'[，。！？,]+', ' ', pro_text)  # 标点转空格
            pro_text = ' '.join(pro_text.split())  # 去除连续空格
        
        return pro_text
   
    def extract_texts(s):
        rec_text = None
        correct_text = None
        
        try:
            # 主方案：只匹配双引号包裹的recognized_text字段
            # 从"recognized_text":"开始，直到下一个", "duplicate_text":"结束
            rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)",\s*"duplicate_text"', s, re.DOTALL)
            if rec_match:
                rec_text = rec_match.group(1)
            
            # 备选方案1：如果没有duplicate_text字段，尝试匹配到下一个字段或对象结束
            if not rec_text:
                rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"(?:,\s*"[^\"]*"|\s*})', s, re.DOTALL)
                if rec_match:
                    rec_text = rec_match.group(1)
            
            # 备选方案2：最基本的匹配，只找字段名和值
            if not rec_text:
                rec_match = re.search(r'"recognized_text"\s*:\s*"(.*?)"', s, re.DOTALL)
                if rec_match:
                    rec_text = rec_match.group(1)
            
            # 对correct_text应用相同的提取逻辑
            correct_match = re.search(r'"correct_text"\s*:\s*"(.*?)"(?:,\s*"[^\"]*"|\s*})', s, re.DOTALL)
            if correct_match:
                correct_text = correct_match.group(1)
            
            if not correct_text:
                correct_match = re.search(r'"correct_text"\s*:\s*"(.*?)"', s, re.DOTALL)
                if correct_match:
                    correct_text = correct_match.group(1)
            
            # 文本清洗：处理转义字符并移除JSON格式内容
            def clean_text(text):
                if text is None:
                    return None
                
                try:
                    # 处理常见的转义字符
                    # 将\n转换为实际的换行符
                    text = text.replace('\\n', '\n')
                    # 将\t转换为实际的制表符
                    text = text.replace('\\t', '\t')
                    # 将\r转换为实际的回车符
                    text = text.replace('\\r', '\r')
                    # 将\\转换为单个反斜杠
                    text = text.replace('\\\\', '\\')
                    # 将\'转换为单引号
                    text = text.replace('\\\'', "'")
                    # 将\"转换为双引号
                    text = text.replace('\\"', '"')
                    
                    # 移除HTML标签或特殊标记
                    text = re.sub(r'<[^>]*>', '', text)
                    
                    return text
                except:
                    return text
            
            # 应用清洗函数
            rec_text = clean_text(rec_text)
            correct_text = clean_text(correct_text)
            
        except Exception as e:
            # 发生异常时返回原始提取的内容
            pass
        
        return rec_text, correct_text
    def pre_c(string, remove_punctuation=False):
        """
        加强版预处理函数：
        - 默认同时去除空白字符和标点符号
        - 可通过参数控制是否去除标点符号
        - 保留中文、英文、数字和#号
        """
        if not string:
            return ""
            
        # 先去除空白字符（保持与原始函数的兼容性）
        result = re.sub(r'\s+', '', string)
        
        # 如果需要去除标点符号
        if remove_punctuation:
            # 去除所有标点符号，保留中文、英文、数字和#号
            punct_pattern = re.compile(r'[^一-龥a-zA-Z0-9#]')
            result = punct_pattern.sub('', result)
            
        return result
    def bad_string_rec(string):
        # 先移除 '<' 和 '>' 字符
        filtered_string = string.replace('<', '').replace('>', '')
        # 使用 pre_c 函数移除所有空白字符
        processed_string = pre_c(filtered_string)
        # 避免除以零的情况
        total_length = len(processed_string) if processed_string else 1
        hash_ratio = string.count('#') / total_length
        return hash_ratio > 0.5, hash_ratio
    # def get_quality_score(rec_text):
    #     # 修改为四舍五入的5分制，处理除零情况
    #     preprocessed = pre_c(rec_text)
        
    #     if not preprocessed:
    #         return 0
    #     score = 5 * (1 - (preprocessed.count('#') / len(preprocessed)))
    #     return max(0, min(5, score))
    def get_quality_score(rec_text):
        # 计算#占比归类为3档
        total_chars = len(pre_c(rec_text))
        hash_count = rec_text.count('#')
        hash_ratio = hash_count / total_chars if total_chars > 0 else 0
        #结构错误率 0.25-1都很糟糕无法接受0分 0.05-0.25属于还算可以接受1分 0-0.05 就是可接受2分 错误率低于1/20
        if hash_ratio<=0.05:
            return 2
        elif hash_ratio<=0.25:
            return 1 
        else:
            return 0
    def contain_chinese(prompts):
        return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False
    def get_semantic_score(rec_text, correct_text, ref_target):
        preprocessed_rec = pre_c(rec_text)
        preprocessed_correct = pre_c(correct_text)
        preprocessed_ref = pre_c(ref_target)
        if max(len(preprocessed_rec), len(preprocessed_ref)) == 0:
            return 0
        
        correct_ratio = (len(preprocessed_correct) / max(len(preprocessed_rec), len(preprocessed_ref)))
        #正确字符率归类为3档 0-0.5 糟糕0分 0.5-0.8 还能接受一些给1分 0.8以上很不错了给2分 区间非均匀是考虑人类评估的可接受性
        if correct_ratio>=0.8:
            return 2
        elif correct_ratio>=0.5:
            return 1
        else:
            return 0
    try:
        # 第一种方式：JSON解析
        answer_dict, answer_keys = parse_json_from_answer(string)
        rec_text_raw = answer_dict.get('recognized_text')
        correct_text_raw = answer_dict.get('correct_text')
    except:
        bad_case,hash_rate =  bad_string_rec(string)
        if ref_target is None and  bad_case:#bad rec and too long for a response
                return None, 0 ,"<###> * N " , None
        rec_text_raw, correct_text_raw = extract_texts(string)
        if rec_text_raw is None:
                # print('rec text is a must for a appropriate response')
                return None, None, "<JSON_ERROR>", "<JSON_ERROR>"
        elif correct_text_raw is None and ref_target is not None:
            # print('correct text is a must for a appropriate reference-based Response')
            rec_text = process_raw_text(rec_text_raw)
            rec_text_cal = rec_text.replace('<#>','#').replace('<###>','###')
            quality_score = get_quality_score(rec_text)
            return 0, quality_score, rec_text_raw, "<JSON_ERROR>"
    
    rec_text = process_raw_text(rec_text_raw)
    correct_text = process_raw_text(correct_text_raw)

    if  'F' in task_name:
        rec_text_cal = rec_text.replace('<#>','#').replace('<###>','###')
        quality_score = get_quality_score(rec_text_cal)
        semantic_score = None
        final_correct_text = None
    else:
        if contain_chinese(ref_target):
            rec_text_cal = rec_text.replace('<#>','#').replace('<###>','###')
            correct_text = correct_text.replace('<#>','#').replace('<###>','###')
            #process_correct
            final_correct_text = ''.join([c for c in correct_text if c != '#' and c in rec_text_cal])
            assert ref_target is not None,'no ref for ref-based task'
            #todo计算得分
            quality_score = get_quality_score(rec_text_cal)
            #todo: semantic score int
            semantic_score = get_semantic_score(rec_text,final_correct_text,ref_target)
        else:
            #en mode
            rec_text_cal = rec_text.replace('<#>','#').replace('<###>','###').upper()
            correct_text = correct_text.replace('<#>','#').replace('<###>','###').upper()
            #process_correct
            # final_correct_text = ''.join([c for c in correct_text if c != '#' and c in rec_text_cal])
            final_correct_text = correct_text
            assert ref_target is not None,'no ref for ref-based task'
            #todo计算得分
            quality_score = get_quality_score(rec_text_cal)
            #todo: semantic score int
            semantic_score = get_semantic_score(rec_text,final_correct_text,ref_target)
            
    return semantic_score,quality_score,rec_text ,final_correct_text
def parse_json_from_answer(answer_raw):
    """
    从原始answer字符串中提取并解析JSON内容，支持处理尾部额外文本
    返回：(解析后的字典, 字典keys列表)
    """
    # 1. 提取可能的JSON内容（处理不同包裹格式）
    json_str = None
    
    # 优先匹配```json包裹的情况
    json_block_pattern = re.compile(r'```json\s*([\s\S]*?)\s*```', re.DOTALL)
    match = json_block_pattern.search(answer_raw)
    if match:
        json_str = match.group(1).strip()
    
    # 若未匹配到，处理单/双引号包裹或无包裹的情况
    if not json_str:
        processed = answer_raw.strip()
        # 去除首尾匹配的单/双引号
        if (processed.startswith("'") and processed.endswith("'")) or \
           (processed.startswith('"') and processed.endswith('"')):
            processed = processed[1:-1].strip()
        json_str = processed
    
    # 检查是否提取到内容
    if not json_str:
        assert False, "未提取到任何可能的JSON内容"
    
    # 2. 关键改进：截断JSON对象/数组后的额外文本（如Note、注释等）
    # 处理JSON对象（以}结尾）
    if '}' in json_str:
        last_brace_idx = json_str.rfind('}')
        json_str = json_str[:last_brace_idx + 1]  # 保留到最后一个}
    # 处理JSON数组（以]结尾）
    elif ']' in json_str:
        last_bracket_idx = json_str.rfind(']')
        json_str = json_str[:last_bracket_idx + 1]  # 保留到最后一个]
    
    # 3. 解析JSON（带错误处理）
    first_e = None
    # 第一次尝试直接解析
    try:
        answer_dict = json.loads(json_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as e:
        first_e = e
    
    # 清理内部换行符（转为JSON兼容的\\n）
    def replace_newlines(match):
        inner = match.group(1)
        return inner.replace('\n', '\\n').replace('\r', '\\r')
    cleaned_str = re.sub(r'"([^"]*)"', lambda m: f'"{replace_newlines(m)}"', json_str)
    
    # 第二次尝试解析清理后的内容
    try:
        answer_dict = json.loads(cleaned_str)
        return answer_dict, list(answer_dict.keys())
    except json.JSONDecodeError as second_e:
        error_msg = [
            f"JSON解析失败（原始内容前50字符: {json_str[:50]}...）",
            f"第一次错误: {str(first_e)}" if first_e else "",
            f"清理后内容前50字符: {cleaned_str[:50]}...",
            f"第二次错误: {str(second_e)}"
        ]
        assert False, "\n".join(filter(None, error_msg))