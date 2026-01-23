"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import numpy as np

from synthtiger import utils
from synthtiger.components.color import RGB
from synthtiger.components.component import Component
from synthtiger.layers import Layer


class TextBorder(Component):
    def __init__(
        self, size=(1, 5), rgb=((0, 255), (0, 255), (0, 255)), alpha=(1, 1), grayscale=0
    ):
        super().__init__()
        self.size = size
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale
        self._color = RGB()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        size = meta.get("size", np.random.randint(self.size[0], self.size[1] + 1))
        rgb = meta.get(
            "rgb",
            (
                np.random.randint(self.rgb[0][0], self.rgb[0][1] + 1),
                np.random.randint(self.rgb[1][0], self.rgb[1][1] + 1),
                np.random.randint(self.rgb[2][0], self.rgb[2][1] + 1),
            ),
        )
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        grayscale = meta.get("grayscale", np.random.rand() < self.grayscale)

        meta = {
            "size": size,
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        size = meta["size"]

        for layer in layers:
            image = layer.output()
            border_image = utils.pad_image(
                image, top=size, right=size, bottom=size, left=size
            )
            border_image = utils.dilate_image(border_image, size)

            border_layer = Layer(border_image)
            border_layer.center = layer.center
            self._color.apply([border_layer], meta)

            out_layer = (layer + border_layer).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta

class AdvancedTextBorder(Component):
    """基于字体大小动态计算边框大小的文本边框组件"""
    def __init__(
        self, size_ratio=(0.05, 0.15), min_size=1, max_size=12,
        rgb=((0, 255), (0, 255), (0, 255)), alpha=(1, 1), grayscale=0
    ):
        super().__init__()
        # 使用size_ratio代替固定的size，范围表示字体大小的比例
        self.size_ratio = size_ratio
        # 设置最小和最大边框大小限制
        self.min_size = min_size
        self.max_size = max_size
        # 颜色和透明度参数
        self.rgb = rgb
        self.alpha = alpha
        self.grayscale = grayscale
        # 颜色应用组件
        self._color = RGB()

    def sample(self, meta=None):
        if meta is None:
            meta = {}
        # 采样边框大小比例
        size_ratio = meta.get(
            "size_ratio",
            np.random.uniform(self.size_ratio[0], self.size_ratio[1])
        )

        # 采样颜色和透明度
        rgb = meta.get(
            "rgb",
            (
                np.random.randint(self.rgb[0][0], self.rgb[0][1] + 1),
                np.random.randint(self.rgb[1][0], self.rgb[1][1] + 1),
                np.random.randint(self.rgb[2][0], self.rgb[2][1] + 1),
            ),
        )
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        grayscale = meta.get("grayscale", np.random.rand() < self.grayscale)

        meta = {
            "size_ratio": size_ratio,
            "rgb": rgb,
            "alpha": alpha,
            "grayscale": grayscale,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        size_ratio = meta["size_ratio"]
    
        # 应用边框效果
        for layer in layers:
            # 从图层属性中获取字体大小
            try:
                font_size = layer.ft_size
                # 计算基于字体大小的边框像素值
                size = int(round(font_size * size_ratio))
                # 确保边框大小在允许的范围内
                size = max(self.min_size, min(self.max_size, size))
            except AttributeError:
                # 如果图层没有ft_size属性，使用默认的最小值
                size = self.min_size
                print("Warning: AdvancedTextBorder.apply() - layer has no ft_size attribute, using minimum size.")

            # 将计算出的size添加到meta中，以便后续使用
            layer_meta = meta.copy()
            layer_meta["size"] = size

            image = layer.output()
            border_image = utils.pad_image(
                image, top=size, right=size, bottom=size, left=size
            )
            border_image = utils.dilate_image(border_image, size)

            border_layer = Layer(border_image)
            border_layer.center = layer.center
            self._color.apply([border_layer], layer_meta)

            out_layer = (layer + border_layer).merge()
            layer.image = out_layer.image
            layer.quad = out_layer.quad

        return meta