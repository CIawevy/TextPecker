"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthtiger import utils
from synthtiger.layers.layer import Layer

#zaozi import
import requests
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import re
from svg.path import parse_path
import random
import numpy as np
import cv2
import math

class Stroke:
    """表示汉字的一个笔画"""
    def __init__(self, path, points, stroke_num, is_in_radical=False):
        self.path = path
        self.points = points  # 笔画的采样点列表 [(x1, y1), (x2, y2), ...]
        self.stroke_num = stroke_num
        self.is_in_radical = is_in_radical

    @property
    def center(self):
        """计算笔画的中心点（所有采样点的平均坐标）"""
        # if not self.points:
        #     return (0.0, 0.0)
        xs, ys = zip(*self.points)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def translate(self, dx, dy):
        """平移笔画的所有点"""
        self.points = [(x + dx, y + dy) for x, y in self.points]

class Character:
    """表示一个汉字"""
    def __init__(self, symbol, strokes):
        self.symbol = symbol
        self.strokes = strokes  # 笔画列表


class ZaoziTextLayer(Layer):
    def __init__(
        self,
        char,
        path,
        size,
        color=(0, 0, 0, 255),
        bold=False,
        vertical=False,
        pre_load_data=None,
        hanzi_list=None,
        method_prob = 1.0,
        char_types = ['delete', 'add', 'self_swap'],
        type_probs = [0.4, 0.4, 0.4], #0.512的正确字符概率
        
        
    ):
        # https://en.wikipedia.org/wiki/Backslash
        char = char.replace("\\", "＼")
        
        #zaozi layer features
        #支持原有textlayer接口，同时加入概率采样的zaozi接口

        self.char_types = char_types
        self.type_probs = type_probs
        self.method_prob = method_prob
        self.sample_density = 8
        self.base_url = "https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/{}.json"

        self.render_success=True
        self.char_data,stat1 = self._try_get_char_data(char,pre_load_data)
        self.debug=False

        if np.random.random() < method_prob and stat1: #zaozi模式 若字符无法获取笔画，就正常渲染到模式二

            
            sample_char = random.choice(hanzi_list)
            while sample_char == char or pre_load_data.get(sample_char,None) is not None:
                sample_char = random.choice(hanzi_list)
                
            self.sample_char_data = pre_load_data.get(sample_char)
        
            image , bbox = self._render_zaozi_text(char,sample_char,size,color)
            super().__init__(image)
            self.bbox = bbox

        else:
            #普通模式
            #todo新增模式三，#2.用zaozi的笔画数据在现有字体的字上加笔画
            font = self._read_font(path, size)
            image, bbox = self._render_text(char, font, color, bold, vertical)
            self.final_char = char

    
            super().__init__(image)
            self.bbox = bbox
      
    
    def _render_zaozi_text(self,char,sample_char,size,color):
        #here font is used only to get char size 
        #采用方案一，无须font，直接用笔画数据构造多样错误，#1.直接用zaozi函数采样正楷手写字体，造字 笔画随机删除， 笔画随机混合， 笔画随机自我交换
        original_char = self.parse_char_data(char, self.char_data)
        sample_char_obj = self.parse_char_data(sample_char, self.sample_char_data) 
        # 随机选择要应用的笔画操作类型
        selected_types = []
        for i, char_type in enumerate(self.char_types):
            if np.random.random() < self.type_probs[i]:
                selected_types.append(char_type)
        modified_char, final_stypes = self._process_strokes(original_char, sample_char_obj, selected_types)
        self.final_char = modified_char.symbol 
        if char != '#' and modified_char.symbol =='#':
            self.final_char = '<#>'
        

        image, bbox = self._get_zaozi_image(modified_char,size,color)
        return image, bbox
    def _get_zaozi_image(self, modified_char, size, color):
        bbox = (0,0,size,size)
        image = self.draw_character_v2(modified_char, canvas_size=(size,size), color=color) #
        image = np.array(image, dtype=np.float32)
        return image, bbox
    def draw_filled_character(self, draw, character, scale, x_offset, y_offset, canvas_height, color):
        """绘制实心汉字（改进版：填充SVG路径闭合区域）"""
        # 先收集所有笔画的点
        all_points = []
        for stroke in character.strokes:
            if len(stroke.points) >= 2:
                # 转换坐标（适配PIL坐标系）
                temp_points = [
                    (round(x * scale + x_offset),
                     round(canvas_height - (y * scale + y_offset)))
                    for x, y in stroke.points
                ]
                # 确保路径闭合（若SVG路径含Z命令，此处会自然闭合）
                if temp_points[0] != temp_points[-1]:
                    temp_points.append(temp_points[0])  # 手动闭合多边形
                all_points.append(temp_points)
        
        if not all_points:
            return
        
        # 1. 填充闭合区域（核心改进：替换线条加粗为多边形填充）
        for points in all_points:
            draw.polygon(points, fill=color)  # 填充路径包围的区域
        
    def calculate_positioning(self, character, canvas_size=None):
        """精确计算缩放和居中参数"""
        canvas_width, canvas_height = canvas_size or self.default_canvas_size
        all_points = []
        for stroke in character.strokes:
            all_points.extend(stroke.points)
        
        if not all_points:
            return 1.0, canvas_width/2, canvas_height/2
        
        # 计算原始边界框
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        char_width = max_x - min_x
        char_height = max_y - min_y
        
        # 预留10%的边距
        padding = min(canvas_width, canvas_height) * 0.1
        available_width = canvas_width - 2 * padding
        available_height = canvas_height - 2 * padding
        
        # 计算缩放比例
        scale = 1.0
        if max(char_width, char_height) > 0:
            scale = min(available_width / char_width, available_height / char_height)
        
        # 计算居中偏移量
        char_center_x = (min_x + max_x) / 2
        char_center_y = (min_y + max_y) / 2
        
        # 适配坐标系转换的偏移量
        x_offset = padding + (available_width / 2) - (char_center_x * scale)
        y_offset = padding + (available_height / 2) - (char_center_y * scale)
        
        return scale, x_offset, y_offset

    def draw_character_v2(self, character, canvas_size=None, color='black'):

        """改进的汉字绘制方法：支持边框、阴影渲染，自动处理3通道RGB颜色"""
        canvas_size = canvas_size or self.default_canvas_size
        image = Image.new("RGBA", canvas_size)
        
        # 计算字符定位参数（缩放+偏移）
        scale, x_offset, y_offset = self.calculate_positioning(character, canvas_size)
        
        # 创建2倍尺寸临时画布（提高渲染精度）
        temp_size = (canvas_size[0] * 2, canvas_size[1] * 2)
        temp_image = Image.new('RGBA', temp_size, color=(255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_image)
        temp_scale = scale * 2
        temp_x_offset = x_offset * 2
        temp_y_offset = y_offset * 2
        
        # 绘制字符主体（实心填充）
        self.draw_filled_character(temp_draw, character, temp_scale, temp_x_offset, temp_y_offset, temp_size[1], color)
        
        # 高质量缩放回原始尺寸
        image = temp_image.resize(canvas_size, Image.LANCZOS)
        return image
        
    def _process_strokes(self, original_char, sample_char, stype_list):
        """内部辅助函数：处理笔画操作（add/delete/swap）"""
        # 复用原逻辑中的笔画处理代码，统一管理操作顺序和参数
        default_ratio_config = {
            'delete_ratio': [0.1, 0.5],
            'add_ratio': [0.1, 0.5],
            'swap_ratio': [0.1, 0.5]
        }
        final_ratio_config = default_ratio_config  # 可根据config覆盖，此处简化

        valid_stypes = ['delete', 'add', 'self_swap']
        sort_priority = {'delete': 0, 'add': 1, 'self_swap': 2}
        sorted_stypes = sorted(list(set(stype_list)), key=lambda x: sort_priority[x])

        current_char = original_char
        final_stypes = []
        
     
        stat = False
        for st in sorted_stypes:
            if st not in valid_stypes:
                continue
            if st == 'delete':
                current_char, stat = self.op_delete(current_char, final_ratio_config['delete_ratio'])
            elif st == 'add' and sample_char:
                current_char, stat = self.op_add(current_char, sample_char, final_ratio_config['add_ratio'])
            elif st == 'self_swap':
                current_char, stat = self.op_self_swap(current_char, final_ratio_config['swap_ratio'])
            

            if stat:
                final_stypes.append(st)
           
        return current_char, final_stypes

    def op_delete(self, character, ratio_range):
        """删除笔画操作（优化版）"""
        original_strokes = character.strokes.copy()  # 浅拷贝（Stroke对象无需深拷贝）
        stroke_count = len(original_strokes)
        
        # 边界条件检查（保持不变）
        if stroke_count <= 1:
            print(f"[op_delete] 笔画数{stroke_count}，无法删除（需≥2画）→ 操作失败")
            return character, False
        
        # 计算删除数量（保持不变）
        min_del = max(1, int(stroke_count * ratio_range[0]))
        max_del = min(stroke_count - 1, int(stroke_count * ratio_range[1]))
        delete_count = random.randint(min_del, max_del)
        delete_idxs = random.sample(range(stroke_count), delete_count)
        
        # 优化1：将delete_idxs转为set，加速成员检查（O(1)）
        delete_idxs_set = set(delete_idxs)
        new_strokes = [s for idx, s in enumerate(original_strokes) if idx not in delete_idxs_set]
        
        # 重新编号（保持不变）
        for new_idx, stroke in enumerate(new_strokes):
            stroke.stroke_num = new_idx
        
        new_character = Character('#', new_strokes)
        # 优化2：调试打印添加条件（通过config控制，默认关闭）
        if self.debug:
            print(f"[op_delete] 成功删除{delete_count}画（索引{delete_idxs}）→ 笔画数{stroke_count}→{len(new_strokes)}")
        return new_character, True

    def op_self_swap(self, character, ratio_range):
        """笔画交换操作（优化版）"""
        original_strokes = character.strokes.copy()  # 浅拷贝
        stroke_count = len(original_strokes)
        
        # 边界条件检查（保持不变）
        if stroke_count < 2:
            print(f"[op_self_swap] 笔画数{stroke_count}，无法交换位置（需≥2画）→ 操作失败")
            return character, False
        
        # 计算交换对数（保持不变）
        min_swap_pairs = max(1, int(stroke_count * ratio_range[0]) // 2)
        max_swap_pairs = max(min_swap_pairs, int(stroke_count * ratio_range[1]) // 2)
        swap_pair_count = random.randint(min_swap_pairs, max_swap_pairs)
        
        # 选择交换索引（保持不变）
        available_idxs = list(range(stroke_count))
        if len(available_idxs) < 2 * swap_pair_count:
            swap_pair_count = len(available_idxs) // 2
            if self.debug:
                print(f"[op_self_swap] 可用笔画不足，调整交换对数为{swap_pair_count}")
        if swap_pair_count == 0:
            print(f"[op_self_swap] 无可用交换对→ 操作失败")
            return character, False
        
        selected_idxs = random.sample(available_idxs, 2 * swap_pair_count)
        swap_pairs = [(selected_idxs[i], selected_idxs[i+1]) for i in range(0, len(selected_idxs), 2)]
        
        # 核心优化：使用numpy向量化平移笔画点
        swapped_strokes = original_strokes.copy()
        for (idx1, idx2) in swap_pairs:
            stroke1 = swapped_strokes[idx1]
            stroke2 = swapped_strokes[idx2]
            
            # 计算偏移量（保持不变）
            dx1, dy1 = (stroke2.center[0] - stroke1.center[0], stroke2.center[1] - stroke1.center[1])
            dx2, dy2 = (stroke1.center[0] - stroke2.center[0], stroke1.center[1] - stroke2.center[1])
            
            # 优化1：向量化平移（替代for循环遍历points）
            # stroke1平移
            points_np = np.array(stroke1.points)
            points_np += [dx1, dy1]  # 向量化操作（比列表推导快5-10倍）
            stroke1.points = points_np.tolist()  # 按需转回列表（若后续流程允许，可直接存numpy数组）
            
            # stroke2平移（同上）
            points_np = np.array(stroke2.points)
            points_np += [dx2, dy2]
            stroke2.points = points_np.tolist()
            
            # 优化2：调试打印开关
            if self.debug:
                print(f"[op_self_swap] 交换笔画{idx1+1}与{idx2+1}位置：")
                print(f"  - 原中心：笔画{idx1+1}({stroke1.center[0]-dx1:.1f},{stroke1.center[1]-dy1:.1f}) → 笔画{idx2+1}({stroke2.center[0]-dx2:.1f},{stroke2.center[1]-dy2:.1f})")
                print(f"  - 新中心：笔画{idx1+1}({stroke1.center[0]:.1f},{stroke1.center[1]:.1f}) → 笔画{idx2+1}({stroke2.center[0]:.1f},{stroke2.center[1]:.1f})")

        new_character = Character('#', swapped_strokes)
        if self.debug:
            print(f"[op_self_swap] 成功交换{swap_pair_count}对笔画位置→ 操作成功")
        return new_character, True

    def op_add(self, character, character_2, ratio_range):
        """添加笔画操作（优化版）"""
        original_strokes = character.strokes
        sample_strokes = character_2.strokes
        orig_count = len(original_strokes)
        sample_count = len(sample_strokes)
        
        if orig_count == 0 or sample_count == 0:
            print(f"[op_add] 原始笔画数{orig_count}或样本笔画数{sample_count}无效→ 操作失败")
            return character, False
        
        # 计算添加数量范围
        min_add = max(1, int(orig_count * ratio_range[0]))
        max_add = min(sample_count, int(orig_count * ratio_range[1]))
        
        # 修复：确保min_add <= max_add
        if min_add > max_add:
            # 当样本字符笔画数不足时，使用样本字符的笔画数作为基准
            min_add = max(1, min(sample_count, 1))  # 至少添加1个或根据样本字符调整
            max_add = min(sample_count, max(min_add, 1))
        
        # 处理min_add等于max_add的情况
        if min_add == max_add:
            add_count = min_add
        else:
            add_count = random.randint(min_add, max_add)
        
        # 确保add_count有效
        add_count = max(1, min(add_count, sample_count))
        
        add_idxs = random.sample(range(sample_count), add_count)
        
        # 后续代码保持不变
        # 优化1：使用set存储添加索引（若后续需检查）
        add_idxs_set = set(add_idxs)
        
        # 优化2：直接创建新列表，避免original_strokes.copy()
        new_strokes = list(original_strokes)  # 浅拷贝原笔画列表
        for add_idx in add_idxs:
            if add_idx not in add_idxs_set:
                continue  # 双重校验（可选）
            sample_stroke = sample_strokes[add_idx]
            # 优化3：使用numpy数组存储points（加速后续平移操作）
            new_stroke = Stroke(
                path=sample_stroke.path,
                points=np.array(sample_stroke.points),  # 存储为numpy数组
                stroke_num=len(new_strokes),
                is_in_radical=False
            )
            new_strokes.append(new_stroke)
        
        new_character = Character('#', new_strokes)
        if self.debug:
            print(f"[op_add] 从样本字添加{add_count}画（索引{add_idxs}）→ 笔画数{orig_count}→{len(new_strokes)}→ 操作成功")
        return new_character, True
    def parse_char_data(self, character, char_json):
        """解析完整路径，获取精确笔画点"""
        if not char_json:
            return None
            
        strokes = []
        rad_strokes = char_json.get('radStrokes', [])
        
        for i, path_str in enumerate(char_json.get('strokes', [])):
            points = []
            try:
                # 解析SVG路径为可采样的对象
                path = parse_path(path_str)
                # 根据路径长度动态调整采样点数量
                path_length = path.length()
                sample_count = max(15, int(path_length * self.sample_density))
                
                # 均匀采样路径上的点
                for t in range(sample_count + 1):
                    param = t / sample_count
                    point = path.point(param)
                    points.append((round(point.real, 2), round(point.imag, 2)))
                
            except Exception as e:
                print(f"解析笔画路径出错: {str(e)}")
                # 备用方案：使用简单正则提取点
                coords = re.findall(r'[-+]?\d+\.?\d*', path_str)
                if len(coords) >= 2:
                    coords = [float(c) for c in coords]
                    points = [(coords[j], coords[j+1]) for j in range(0, len(coords)-1, 2)]
            
            is_in_radical = i in rad_strokes
            stroke = Stroke(path_str, points, i, is_in_radical)
            strokes.append(stroke)
        
        return Character(character, strokes)
    def _fetch_hanzi_data(self, char):
        """从CDN获取汉字笔画数据"""
        url = self.base_url.format(char)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            
            # # 尝试备用URL
            # alternative_url = f"https://cdn.jsdelivr.net/npm/hanzi-writer-data@latest/data/{char}.json"
            # alt_response = requests.get(alternative_url, timeout=10)
            # if alt_response.status_code == 200:
            #     return alt_response.json()
                
            print(f"获取汉字'{char}'数据失败")
            return None
        except Exception as e:
            print(f"获取数据出错: {str(e)}")
            return None
    def _try_get_char_data(self,char,pre_load_data=None):
        """
        Check if the char exists in the src
        """
         # 获取汉字原始数据
        if pre_load_data is not None:
            char_data = pre_load_data.get(char,None)
        else:
            char_data = self._fetch_hanzi_data(char)
        stat = True if char_data is not None else False
        return char_data ,stat
    def _render_text(self, text, font, color, bold, vertical):
        if not vertical:
            image, bbox = self._render_hori_text(text, font, color, bold)
        else:
            image, bbox = self._render_vert_text(text, font, color, bold)

        return image, bbox

    def _get_image(self, text, font, color, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)
        bbox = self._get_bbox(text, font, vertical)
        width, height = bbox[2:]

        image = Image.new("RGBA", (width, height))
        draw = ImageDraw.Draw(image)
        draw.text(
            (0, 0),
            text,
            fill=color,
            font=font,
            stroke_width=stroke_width,
            direction=direction,
        )
        image = np.array(image, dtype=np.float32)

        return image, bbox
    def _get_text_size(self,font, text_str):
        try:
            _, _, w, h = font.getbbox(text_str)
        except AttributeError:
            w, h = font.getsize(text_str)  # Pillow<8

        return w, h
    def _get_bbox(self, text, font, vertical):
        # 完全移除direction参数，避免libraqm依赖
        
        if not vertical:
            ascent, descent = font.getmetrics()
            

            # 尝试使用getbbox获取宽度
            left, top, right, bottom = font.getbbox(text)
            width = right - left
        
                
            height = ascent + descent
            bbox = [0, -ascent, width, height]
        else:
  
            left, top, right, bottom = font.getbbox(text)
            width = right - left
            height = bottom  # 直接使用bottom值作为高度
        
                
            bbox = [-width // 2, 0, width, height]

        return bbox
    # def _get_bbox(self, text, font, vertical):
    #     direction = self._get_direction(vertical)

    #     if not vertical:
    #         ascent, descent = font.getmetrics()
    #         width = font.getsize(text, direction=direction)[0]
    #         height = ascent + descent
    #         bbox = [0, -ascent, width, height]
    #     else:
    #         width, height = font.getsize(text, direction=direction)
    #         bbox = [-width // 2, 0, width, height]

    #     return bbox

    def _get_inner_bbox(self, text, font, bold, vertical):
        stroke_width = self._get_stroke_width(bold)
        direction = self._get_direction(vertical)

        mask, offset = font.getmask2(
            text, stroke_width=stroke_width, direction=direction
        )
        bbox = mask.getbbox()
        left = max(bbox[0] + offset[0], 0)
        top = max(bbox[1] + offset[1], 0)
        right = max(bbox[2] + offset[0], 0)
        bottom = max(bbox[3] + offset[1], 0)
        width = max(right - left, 0)
        height = max(bottom - top, 0)
        bbox = [left, top, width, height]

        return bbox

    def _get_stroke_width(self, bold):
        stroke_width = int(bold)
        return stroke_width

    def _get_direction(self, vertical):
        direction = "ltr" if not vertical else "ttb"
        return direction
    def _read_font(self, path, size):
        font = ImageFont.truetype(path, size=size)
        return font

    
    def _render_hori_text(self, text, font, color, bold):
        image, bbox = self._get_image(text, font, color, bold, False)
        return image, bbox

    def _render_vert_text(self, text, font, color, bold):
        chars = utils.split_text(text, reorder=True)
        patches = []
        bboxes = []

        for char in chars:
            patch, bbox = self._render_vert_char(char, font, color, bold)
            patches.append(patch)
            bboxes.append(bbox)

        width = max([bbox[2] for bbox in bboxes])
        height = sum([bbox[3] for bbox in bboxes])
        left = min([bbox[0] for bbox in bboxes])
        bottom = 0

        for bbox in bboxes:
            bbox[0] -= left
            bbox[1] = bottom
            bottom += bbox[3]

        image = utils.create_image((width, height))
        for patch, (x, y, w, h) in zip(patches, bboxes):
            image[y : y + h, x : x + w] = patch

        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_char(self, char, font, color, bold):
        fullwidth_char = utils.to_fullwidth(char)[0]

        if utils.vert_orient(fullwidth_char) != "Tr" and fullwidth_char.isalnum():
            return self._render_vert_upright_char(char, font, color, bold)

        if utils.vert_rot_flip(fullwidth_char):
            return self._render_vert_rot_flip_char(char, font, color, bold)

        if utils.vert_right_flip(fullwidth_char):
            return self._render_vert_right_flip_char(char, font, color, bold)

        if utils.vert_orient(fullwidth_char) in ("R", "Tr"):
            return self._render_vert_rot_char(char, font, color, bold)

        return self._render_vert_upright_char(char, font, color, bold)

    def _render_vert_upright_char(self, char, font, color, bold):
        vertical = len(char) <= 1
        image, bbox = self._get_image(char, font, color, bold, vertical)
        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]
        return image, bbox

    def _render_vert_rot_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)
        image, _ = utils.fit_image(image, left=False, right=False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_rot_flip_char(self, char, font, color, bold):
        image, bbox = self._get_image(char, font, color, bold, False)

        ascent, width = -bbox[1], bbox[2]
        left = max(ascent - width, 0) // 2
        right = max(ascent - width, 0) - left
        image = np.pad(image, ((0, 0), (left, right), (0, 0)))
        image = np.rot90(image, k=-1)
        image = np.fliplr(image)

        height, width = image.shape[:2]
        bbox = [-width // 2, 0, width, height]

        return image, bbox

    def _render_vert_right_flip_char(self, char, font, color, bold):
        bbox = self._get_bbox(char, font, False)
        inner_bbox = self._get_inner_bbox(char, font, bold, False)
        sx, sy, patch_width, patch_height = inner_bbox

        patch, _ = self._get_image(char, font, color, bold, False)
        patch = patch[sy : sy + patch_height, sx : sx + patch_width]
        patch_height, patch_width = patch.shape[:2]

        ascent = -bbox[1]
        width, height = max(ascent, patch_width), max(ascent, patch_height)
        dx, dy = max(width - patch_width, 0), max(height - patch_height - sy, 0)

        image = utils.create_image((width, height))
        image[dy : dy + patch_height, dx : dx + patch_width] = patch
        bbox = [-width // 2, 0, width, height]

        return image, bbox

 

class Stroke:
    """表示汉字的一个笔画"""
    def __init__(self, path, points, stroke_num, is_in_radical=False):
        self.path = path
        self.points = points  # 笔画的采样点列表 [(x1, y1), (x2, y2), ...]
        self.stroke_num = stroke_num
        self.is_in_radical = is_in_radical

    @property
    def center(self):
        """计算笔画的中心点（所有采样点的平均坐标）"""
        # if not self.points:
        #     return (0.0, 0.0)
        xs, ys = zip(*self.points)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def translate(self, dx, dy):
        """平移笔画的所有点"""
        self.points = [(x + dx, y + dy) for x, y in self.points]
class Character:
    """表示一个汉字"""
    def __init__(self, symbol, strokes):
        self.symbol = symbol
        self.strokes = strokes  # 笔画列表