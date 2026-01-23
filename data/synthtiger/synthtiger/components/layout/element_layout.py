"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger.components.component import Component


class ElementLayout(Component):
    def __init__(
        self,
        space_horizontal=(0, 0),
        space_vertical=(0, 0),
        length_ratio=(0.5, 0.8),
        ltr=True,
        ttb=True,
        random_offset_range=(0, 0),
        length_ratio_range=(0.5, 0.8),
        offset_prob=0.5,
        margin=0,  # 添加margin参数，默认为0
    ):
        super().__init__()
        self.space_horizontal = space_horizontal
        self.space_vertical = space_vertical
        self.length_ratio = length_ratio
        self.ltr = ltr
        self.ttb = ttb
        self.random_offset_range = random_offset_range
        self.length_ratio_range = length_ratio_range
        self.offset_prob = offset_prob
        self.margin = margin  # 保存margin参数

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        # 采样水平和垂直间距
        space_horizontal = meta.get("space_horizontal", np.random.randint(
            self.space_horizontal[0], self.space_horizontal[1] + 1))
        space_vertical = meta.get("space_vertical", np.random.randint(
            self.space_vertical[0], self.space_vertical[1] + 1))
        # 采样长度比例
        length_ratio_x = np.random.uniform(
            self.length_ratio_range[0], self.length_ratio_range[1])
        length_ratio_y = np.random.uniform(
            self.length_ratio_range[0], self.length_ratio_range[1])
        ltr = meta.get("ltr", self.ltr)
        ttb = meta.get("ttb", self.ttb)

        meta = {
            "space_horizontal": space_horizontal,
            "space_vertical": space_vertical,
            "ltr": ltr,
            "ttb": ttb,
            "length_ratio_x": length_ratio_x,
            "length_ratio_y": length_ratio_y,
        }

        return meta

    def init_layout(self, size, meta=None):
        """初始化布局参数和状态"""
        if meta is None:
            meta = {}
        
        # 从meta中获取size参数
        if size is None:
            raise ValueError("Size parameter is required")
        
        # 采样其他参数
        sampled_meta = self.sample(meta)
        ltr = sampled_meta["ltr"]
        ttb = sampled_meta["ttb"]
        space_horizontal = sampled_meta["space_horizontal"]
        space_vertical = sampled_meta["space_vertical"]
        length_ratio_x = sampled_meta["length_ratio_x"]
        length_ratio_y = sampled_meta["length_ratio_y"]
        
        # 计算子画布尺寸时考虑margin
        sub_width = int(size[0] * length_ratio_x) - 2 * self.margin
        sub_height = int(size[1] * length_ratio_y) - 2 * self.margin
        
        # 随机选择子画布在原画布中的位置，考虑margin
        sub_x = np.random.randint(self.margin, size[0] - sub_width - self.margin + 1)
        sub_y = np.random.randint(self.margin, size[1] - sub_height - self.margin + 1)
        
        # 初始化布局状态
        layout_state = {
            "size": size,
            "sub_width": sub_width,
            "sub_height": sub_height,
            "sub_x": sub_x,
            "sub_y": sub_y,
            "space_horizontal": space_horizontal,
            "space_vertical": space_vertical,
            "ltr": ltr,
            "ttb": ttb,
            "current_x": 0,
            "current_y": 0,
            "line_bottom": 0,
            "is_first_in_line": True
        }
        
        return layout_state

    def try_add_element(self, layer, layout_state):
        """尝试添加单个元素，如果成功则调整topleft并返回True，否则返回False"""
        # 获取元素尺寸
        layer_width = layer.width
        layer_height = layer.height
        
        # 从布局状态中获取参数
        sub_width = layout_state["sub_width"]
        sub_height = layout_state["sub_height"]
        space_horizontal = layout_state["space_horizontal"]
        space_vertical = layout_state["space_vertical"]
        current_x = layout_state["current_x"]
        current_y = layout_state["current_y"]
        line_bottom = layout_state["line_bottom"]
        is_first_in_line = layout_state["is_first_in_line"]
        
        # 计算当前行需要的总宽度
        total_required_width = current_x + layer_width
        if not is_first_in_line:
            total_required_width += space_horizontal
        
        # 基本放置位置（不使用任何偏移）
        base_x, base_y = current_x, current_y
        
        # 标记是否成功放置
        placed = False
        
        # 检查不偏移情况下能否放在当前行
        if is_first_in_line or total_required_width <= sub_width:
            # 当前行可以放置
            # 非行首元素需要加上水平间距
            if not is_first_in_line:
                base_x += space_horizontal
            placed = True
        else:
            # 当前行放不下，检查换行
            new_line_y = line_bottom + space_vertical 
            if new_line_y + layer_height <= sub_height:
                # 换行可以放置
                base_x, base_y = 0, new_line_y
                is_first_in_line = True
                placed = True
            else:
                # 换行也放不下，返回False
                return False
        
        # 如果成功放置，处理偏移
        if placed:
            # 最终放置位置
            final_x, final_y = base_x, base_y
            
            # 处理X轴偏移（仅对新行的首个元素应用，基于prob）
            if is_first_in_line and np.random.random() < self.offset_prob:
                max_x_offset = min(sub_width - layer_width, self.random_offset_range[1])
                min_x_offset = max(0, self.random_offset_range[0])
                if max_x_offset >= min_x_offset:
                    final_x += np.random.randint(min_x_offset, max_x_offset + 1)
            
            # 处理Y轴偏移（在宽度足够的情况下应用，基于prob）
            if np.random.random() < self.offset_prob:
                max_y_offset = min(sub_height - final_y - layer_height, self.random_offset_range[1])
                min_y_offset = max(-final_y, self.random_offset_range[0])
                if max_y_offset >= min_y_offset:
                    final_y += np.random.randint(min_y_offset, max_y_offset + 1)
            
            # 设置元素位置并检查边界
            # 理论上前面的检查应该确保元素可以放置，但保留此检查作为最后的安全保障
            layer.topleft = (final_x, final_y)
            if layer.right <= sub_width and layer.bottom <= sub_height:
                # 将子画布中的元素位置映射到原画布
                layer.left += layout_state["sub_x"]
                layer.top += layout_state["sub_y"]
                
                # 调整方向
                if not layout_state["ltr"]:
                    layer.right = layout_state["size"][0] - layer.left
                if not layout_state["ttb"]:
                    layer.bottom = layout_state["size"][1] - layer.top
                
                # 更新布局状态
                layout_state["current_x"] = layer.right - layout_state["sub_x"]  # 转回子画布坐标
                layout_state["current_y"] = final_y  # 保持当前行的Y坐标（子画布坐标）
                layout_state["line_bottom"] = max(layer.bottom - layout_state["sub_y"] + layout_state["space_vertical"], line_bottom)  # 转回子画布坐标
                layout_state["is_first_in_line"] = False  # 不再是行首元素
                
                return True
        
        return False

    def apply(self, element_layers, meta=None):
        """重新设计的apply方法，支持元素布局"""
        
        # 获取meta参数
        if meta is None:
            meta = {}
        
        # 从meta中获取size参数
        size = meta.get('size')
        if size is None:
            raise ValueError("Size parameter is required in meta")
        
        # 初始化布局
        layout_state = self.init_layout(size, meta)
        
        valid_layers = []
        valid_indices = []
        
        # 尝试添加每个元素
        for idx, layer in enumerate(element_layers):
            # 创建图层副本以避免修改原始图层
            layer_copy = layer.copy()
            
            # 尝试添加元素
            if self.try_add_element(layer_copy, layout_state):
                valid_layers.append(layer_copy)
                valid_indices.append(idx)
        
        # 返回有效的元素图层和对应的索引
        return valid_layers, valid_indices