"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import json

import numpy as np
from PIL import Image, ImageOps

from synthtiger import utils
from synthtiger.components.component import Component


class AdvancedTexture(Component):
    def __init__(self, json_path="", alpha=(1, 1), grayscale=0, crop=0):
        super().__init__()
        # 使用json_path替代原来的paths和weights
        self.json_path = json_path
        self.alpha = alpha
        self.grayscale = grayscale
        self.crop = crop
        self._paths = []
        self._update_paths()

    def sample(self, meta=None):
        if meta is None:
            meta = {}

        if len(self._paths) == 0:
            raise RuntimeError("No texture paths loaded from JSON file")

        # 从JSON加载的路径中随机选择一个
        path = meta.get("path", self._sample_texture())
        alpha = meta.get("alpha", np.random.uniform(self.alpha[0], self.alpha[1]))
        grayscale = meta.get("grayscale", np.random.rand() < self.grayscale)
        crop = meta.get("crop", np.random.rand() < self.crop)

        width, height = self._get_size(path)
        w = meta.get("w", np.random.randint(1, width + 1) if crop else width)
        h = meta.get("h", np.random.randint(1, height + 1) if crop else height)
        x = meta.get("x", np.random.randint(0, width - w + 1) if crop else 0)
        y = meta.get("y", np.random.randint(0, height - h + 1) if crop else 0)

        meta = {
            "path": path,
            "alpha": alpha,
            "grayscale": grayscale,
            "crop": crop,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        }

        return meta

    def apply(self, layers, meta=None):
        meta = self.sample(meta)
        texture = self.data(meta)

        for layer in layers:
            height, width = layer.image.shape[:2]
            image = utils.resize_image(texture, (width, height))
            layer.image = utils.blend_image(image, layer.image, mask=True)

        return meta

    def data(self, meta):
        x, y, w, h = meta["x"], meta["y"], meta["w"], meta["h"]
        texture = self._read_texture(meta["path"], meta["grayscale"])
        texture = texture[y : y + h, x : x + w, ...]
        texture[..., 3] *= meta["alpha"]
        return texture

    def _update_paths(self):
        self._paths = []

        # 如果json_path为空字符串，不执行任何操作
        if not self.json_path:
            return

        # 只有当json_path不为空且文件不存在时，才抛出异常
        if not os.path.exists(self.json_path):
            raise RuntimeError(f"JSON file not found: {self.json_path}")

        # 从JSON文件加载纹理路径
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self._paths = data
            

    def _read_texture(self, path, grayscale=False):
        texture = Image.open(path)
        texture = ImageOps.exif_transpose(texture)
        if grayscale:
            texture = texture.convert("L")
        texture = texture.convert("RGBA")
        texture = np.array(texture, dtype=np.float32)
        return texture

    def _get_size(self, path):
        texture = Image.open(path)
        width, height = texture.size
        exif = dict(texture.getexif())
        if exif.get(0x0112, 1) >= 5:
            width, height = height, width
        return width, height

    def _sample_texture(self):
        # 从加载的路径列表中随机选择一个
        idx = np.random.randint(len(self._paths))
        path = self._paths[idx]
        return path