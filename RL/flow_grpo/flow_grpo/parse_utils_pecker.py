import re
import json
from typing import List
from scipy.optimize import linear_sum_assignment
import numpy as np
import re
import json
import numpy as np
from io import BytesIO
from typing import List
from collections import Counter
from PIL import Image
from scipy.optimize import linear_sum_assignment
from collections import Counter
def get_score_v2(string,ref_target=None,qua_amplify_factor=1.0,vis_cls=False):
    #PARSE INFO
    base_info={
        'ref_target':ref_target,
        'recognized_text':"",
    }
    try:
        # 第一种方式：JSON parser/ complete json format with be extracted
        answer_dict, answer_keys = parse_json_from_answer(string)
        rec_text_raw = answer_dict.get('recognized_text')
        # correct_text_raw = answer_dict.get('correct_text')
    except:
        # bad_case,hash_rate =  bad_string_rec(string)
        # if ref_target is None and  bad_case:#bad rec and too long (repeated) for a response
        #     return 0, 0 ,0 ,"<###> * N " , None
        rec_text_raw = extract_rec_texts(string)
        if rec_text_raw is None and ref_target is None:
            base_info.update({'recognized_text': '<EXTRACT FAILED>'}) #debug
            return 0, 'None', base_info
    
  
    rec_text = process_raw_text(rec_text_raw)   #long string rec_text seperate by " "
    quality_score = get_scaled_quality_score(rec_text,amplify_factor=qua_amplify_factor)
    gned_score = 'None'
    base_info.update({'recognized_text': rec_text})
    #MATCHING
    # 优化后的代码
    if ref_target is not None:
        gned_score, cls_info = get_gned_score(rec_text, ref_target, cls=vis_cls)
        base_info.update(cls_info)  # 直接更新，因为cls_info总是字典
    
    # print(base_info)
    # print(f'quality_score:{quality_score} gned_score:{gned_score}')
    return quality_score, gned_score, base_info
def to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert PIL Image to RGB mode, handling RGBA with white background."""
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")
def px_to_rel(coord, W, H):
    x, y = coord
    x = max(0, min(x, W-1))  # 边界保护
    y = max(0, min(y, H-1))
    rel_x = int(round(x/(W-1)*1000)) if W>1 else 0  # 转整数
    rel_y = int(round(y/(H-1)*1000)) if H>1 else 0
    return rel_x, rel_y

def process_box_input(ori_bbox, image_path):
    """将像素坐标转为相对坐标（0-1000范围）"""
    xmin_px, ymin_px, xmax_px, ymax_px = ori_bbox

    with Image.open(image_path) as img:
        W, H = img.size  # W=图像宽度，H=图像高度

    xmin_rel, ymin_rel = px_to_rel((xmin_px, ymin_px), W, H)
    xmax_rel, ymax_rel = px_to_rel((xmax_px, ymax_px), W, H)
    normalized_bbox = f"bbox_2d:[{xmin_rel}, {ymin_rel}, {xmax_rel}, {ymax_rel}]"

    return normalized_bbox
def contain_chinese(prompts):
    return  True if any('\u4e00' <= c <= '\u9fff' for c in prompts) else False
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
def read_json_file(file_path):
    """读取 JSON 文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def split_chinese_or_mix(text_list):
        def is_chinese(char):
            return '\u4e00' <= char <= '\u9fff'
        new_list = []
        for word in text_list:#could be chinese chars and single english  word or mix
            # 鱼龙混杂
            # fishes
            # 鱼龙fish
            if  not contain_chinese(word):
                new_list.append(word)
            else:
                is_en = False
                mix = False
                split_en_word = ''
                for char in word:
                    if is_chinese(char):
                        is_en=False
                        new_list.append(char)
                    else:#for mix 
                        mix = True
                        is_en = True
                        split_en_word += char
                    if mix and not is_en:
                        new_list.append(split_en_word)
                        split_en_word=''
                        mix=False
        return new_list
def matching_based_nled(gt_list, test_list):
        new_test_list = []
        for _ in test_list:
            new_test_list.extend(_.split(" ", -1))
        test_list = new_test_list

        len_gt, len_test = len(gt_list), len(test_list)
        if len_gt == 0 and len_test == 0:
            return 0.0  # 两者均为空时无误差

        cost_matrix = np.zeros((len_test, len_gt))
        for i, test_item in enumerate(test_list):
            for j, gt_item in enumerate(gt_list):
                cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_cost = cost_matrix[row_ind, col_ind].sum()
        unmatched_penalty = abs(len_test - len_gt)
        #一个改进是直接算unmatched matrix 去算M+N-2I ，I是match的对角阵 部署在v2
        total_cost = matched_cost + unmatched_penalty

        max_len = max(len_gt, len_test)
        normalized_gned = total_cost / max_len
        return normalized_gned
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
        


def extract_rec_texts(s):
    rec_text = None
    # correct_text = None
    
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
        
        # # 对correct_text应用相同的提取逻辑
        # correct_match = re.search(r'"correct_text"\s*:\s*"(.*?)"(?:,\s*"[^\"]*"|\s*})', s, re.DOTALL)
        # if correct_match:
        #     correct_text = correct_match.group(1)
        
        # if not correct_text:
        #     correct_match = re.search(r'"correct_text"\s*:\s*"(.*?)"', s, re.DOTALL)
        #     if correct_match:
        #         correct_text = correct_match.group(1)
        
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
        # correct_text = clean_text(correct_text)
        
    except Exception as e:
        # 发生异常时返回原始提取的内容
        pass
    
    return rec_text
def pre_c(string):
    # 使用正则表达式替换所有空白字符，包括空格和换行符
    return re.sub(r'\s+', '', string)

def process_chinese(gt_chinese, test_chinese):
    """处理中文数据（仅基础分类，无typo判断）"""
    target_dict = Counter(gt_chinese)
    pred_dict = Counter(test_chinese)
    results = {
        'correct_text': [],
        'duplicate_text': [],
        'irrelevant_text': [],
        'typo_text': [],  # 中文无typo，始终为空
        'missing_text': []
    }
    
    for k, v in pred_dict.items():
        if k in target_dict:
            target_freq = target_dict[k]
            if v > target_freq:
                results['duplicate_text'].extend([k] * (v - target_freq))
                results['correct_text'].extend([k] * target_freq)
            elif v == target_freq:
                results['correct_text'].extend([k] * v)
            else:
                results['correct_text'].extend([k] * v)
                results['missing_text'].extend([k] * (target_freq - v))
            del target_dict[k]
        else:
            results['irrelevant_text'].extend([k] * v)
    
    for k, freq in target_dict.items():
        results['missing_text'].extend([k] * freq)
    
    return results

def process_english(gt_english, test_english):
    """处理英文数据（基础分类 + typo识别）"""
    target_dict = Counter(gt_english)
    pred_dict = Counter(test_english)
    results = {
        'correct_text': [],
        'duplicate_text': [],
        'irrelevant_text': [],
        'typo_text': [],
        'missing_text': []
    }
    
    # 基础分类
    for k, v in pred_dict.items():
        if k in target_dict:
            target_freq = target_dict[k]
            if v > target_freq:
                results['duplicate_text'].extend([k] * (v - target_freq))
                results['correct_text'].extend([k] * target_freq)
            elif v == target_freq:
                results['correct_text'].extend([k] * v)
            else:
                results['correct_text'].extend([k] * v)
                results['missing_text'].extend([k] * (target_freq - v))
            del target_dict[k]
        else:
            results['irrelevant_text'].extend([k] * v)
    
    for k, freq in target_dict.items():
        results['missing_text'].extend([k] * freq)
    
    # 识别typo（基于索引精确删除）
    # 记录候选词及其在results中的索引
    target_candidates = [
        (w, idx) for idx, w in enumerate(results['missing_text'])
    ]
    pred_candidates = [
        (w, idx) for idx, w in enumerate(results['irrelevant_text'])
    ]
    
    len_gt, len_test = len(target_candidates), len(pred_candidates)
    if len_gt > 0 and len_test > 0:
        # 计算代价矩阵
        cost_matrix = np.zeros((len_test, len_gt))
        for i, (test_item, _) in enumerate(pred_candidates):
            for j, (gt_item, _) in enumerate(target_candidates):
                cost_matrix[i][j] = normalized_edit_distance(test_item, gt_item)
        
        # 匹配最小代价对
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 筛选typo并记录索引
        typo_indices = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i][j] < 0.5:
                pred_idx = pred_candidates[i][1]
                target_idx = target_candidates[j][1]
                typo_indices.append((pred_idx, target_idx))
        
        # 精确删除
        if typo_indices:
            # 删除irrelevant_text中匹配的typo
            pred_indices_to_remove = sorted([p for p, t in typo_indices], reverse=True)
            for idx in pred_indices_to_remove:
                del results['irrelevant_text'][idx]
            
            # 删除missing_text中匹配的原始词
            target_indices_to_remove = sorted([t for p, t in typo_indices], reverse=True)
            for idx in target_indices_to_remove:
                del results['missing_text'][idx]
            
            # 添加到typo_text
            results['typo_text'].extend([
                pred_candidates[i][0] for i in row_ind
                if cost_matrix[i, col_ind[row_ind == i][0]] < 0.5
            ])
    
    return results

def classify_results(gt_list, test_list):
    """主函数：分离中英文，分别处理后合并结果"""
    # 1. 分离中文和英文数据
    gt_chinese = [w for w in gt_list if contain_chinese(w)]
    gt_english = [w for w in gt_list if not contain_chinese(w)]
    test_chinese = [w for w in test_list if contain_chinese(w)]
    test_english = [w for w in test_list if not contain_chinese(w)]
    
    # 2. 分别处理中文和英文
    chinese_results = process_chinese(gt_chinese, test_chinese)
    english_results = process_english(gt_english, test_english)
    
    # 3. 合并结果（中文结果 + 英文结果）
    final_results = {
        key: chinese_results[key] + english_results[key]
        for key in chinese_results.keys()
    }
    #将value 里的 list转化成 ' ' 连接的string 方便后续json保存
    for key, value in final_results.items():
        final_results[key] = ' '.join(value)
    return final_results



    
def get_gned_score(rec_text,ref_target,cls=False):
    rec_text = re.sub(r'[<>]', '', rec_text)
    if '#' not in ref_target:
        pass
    elif '$' not in ref_target:
        rec_text = rec_text.replace('#','$')
    elif '&' in ref_target:
        rec_text = rec_text.replace('#','&')
    elif '￥' in ref_target:
        rec_text = rec_text.replace('#','￥')

    rec_text_list = rec_text.split(" ")
    ref_target_list = ref_target.split(" ")
    rec_text_list = [_.lower() for _ in rec_text_list]
    ref_target_list = [_.lower() for _ in ref_target_list]
    rec_text_list = split_chinese_or_mix(rec_text_list)
    ref_target_list = split_chinese_or_mix(ref_target_list)

    gned = matching_based_nled(ref_target_list, rec_text_list)
    # gned = matching_based_nled_v2(ref_target_list, rec_text_list)
    cls_results={}
    if cls:
        cls_results = classify_results(ref_target_list, rec_text_list)
    reward = 1-gned
    return reward,cls_results
def get_scaled_quality_score(rec_text, amplify_factor=1.0):
    #  N_P：生成文本的总字符数
    processed_text = pre_c(rec_text)  # 预处理文本（你定义的pre_c函数）
    N_P = len(processed_text)
    if N_P == 0:
        return 0.0  # 无字符时得分0
    
    #  N_a：异常字符数（这里是'#'的数量）
    N_a = rec_text.count('#')
    
    # 异常字符占比
    anomaly_ratio = N_a / N_P if N_P > 0 else 0.0
    
    # 计算基础得分 +约束在[0,1]区间）
    raw_score = 1 - (anomaly_ratio * amplify_factor)
    scaled_score = max(0.0, min(raw_score, 1.0)) 
    
    return scaled_score
    
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
