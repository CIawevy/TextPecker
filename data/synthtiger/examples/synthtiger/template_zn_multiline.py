"""
SynthTIGER
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import math
import cv2
import numpy as np
from PIL import Image
import json
import os
from PIL import Image
import time
import numpy as np
from PIL import Image, ImageDraw
           
from synthtiger import components, layers, templates, utils
MIN_PIXELS = 768*28*28
MAX_PIXELS = 1024*28*28
IMAGE_FACTOR = 28
MAX_RATIO = 200

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]
def save_img( ndarray_img, path='/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/synthtiger/examples/synthtiger/debug/temp.png'):
    img = Image.fromarray(ndarray_img)
    img.save(path)
# 辅助函数
def round_by_factor(x, factor):
    return round(x / factor) * factor

def floor_by_factor(x, factor):
    return math.floor(x / factor) * factor

def ceil_by_factor(x, factor):
    return math.ceil(x / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

class SynthTigerZH(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        
        self.mask_output = config.get("mask_output", True)
        # self.vertical = config.get("vertical", False)
        self.vertical_prob = config.get("vertical_prob", 0.1)
        self.quality = config.get("quality", [95, 95])
        self.visibility_check = config.get("visibility_check", False)
        self.foreground_mask_pad = config.get("foreground_mask_pad", 0)
        self.corpus = components.ChineseCorpus(**config.get("corpus", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        # 使用专门的bg_texture配置初始化背景纹理组件
        self.bg_texture = components.Switch(
            components.AdvancedTexture(), **config.get("bg_texture", {})
        )
        # 保留原有的texture配置用于文本元素
        self.texture = components.Switch(
            components.AdvancedTexture(), **config.get("texture", {})
        )
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {}),
        )
        self.element_layout = components.ElementLayout(**config["element_layout"])
        self.style = components.Switch(
            components.Selector(
                [
                    components.AdvancedTextBorder(),
                    components.TextShadow(),
                    components.TextExtrusion(),
                ]
            ),
            **config.get("style", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )
        self.element_nums = config.get("element_nums", [3, 10])
        self.pre_load_json_path = config.get("pre_load_json_path", '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/hanzi.json')
        self.hanzi_txt_path = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/code/hanzi_statics/现代汉语通用字7000.txt'
        with open(self.pre_load_json_path, 'r', encoding='utf-8') as f:
            self.pre_load_data = json.load(f)
        with open(self.hanzi_txt_path, 'r', encoding='utf-8') as f:
            self.hanzi_list = [line.strip() for line in f if line.strip()]
        self.zaozi_config = config.get("zaozi", {})
  


       
    def generate(self):
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)


        # 1. 首先生成背景图层并获取尺寸
        # 根据配置参数生成背景尺寸
        bg_size = self._generate_background_size()
        # 生成背景图像 从bg image crop指定size到bg layer
        bg_image = self._generate_background_v2(bg_size)
        # # 获取背景图像的高度和宽度
        bg_h, bg_w = bg_image.shape[:2]  # NumPy数组形状为 (height, width, channels)
        
        # # 确保传递给_generate_multiple_text的尺寸是 (width, height) 格式
        size = (bg_w, bg_h)  # 转换为 (width, height)
        #循环采样布局
        

        final_image, font_image, bboxes, font_infos, transform_types, labels = self._generate_multiple_text(size,bg_image.copy())

        # image = _blend_images(fg_image, bg_image, self.visibility_check)
        # image, fg_image, = self._postprocess_images( #加入噪声各种后处理
        #     [image, fg_image]
        # )

        data = {
            "image": final_image,
            "labels": labels,
            "quality": quality,
            "mask": font_image[..., 3],
            "bboxes": bboxes,
            'font_infos': font_infos,
            'transforms_type':transform_types,
        }

        return data
    def _generate_background_size(self):
        # 生成背景尺寸，返回 (width, height) 格式
        min_size = 512
        max_size = 1024
        width = np.random.randint(min_size, max_size + 1)
        height = np.random.randint(min_size, max_size + 1)
        # 使用smart_resize确保尺寸符合要求
        h, w = smart_resize(height, width)  # smart_resize返回 (height, width)
        return (w, h)  # 转换为 (width, height) 返回
        
    # def init_save(self, root):
    #     os.makedirs(root, exist_ok=True)

    #     gt_path = os.path.join(root, "gt.txt")
    #     coords_path = os.path.join(root, "coords.txt")
    #     glyph_coords_path = os.path.join(root, "glyph_coords.txt")

    #     self.gt_file = open(gt_path, "w", encoding="utf-8")
    #     if self.coord_output:
    #         self.coords_file = open(coords_path, "w", encoding="utf-8")
    #     if self.glyph_coord_output:
    #         self.glyph_coords_file = open(glyph_coords_path, "w", encoding="utf-8")
    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

        # 只创建一个jsonl文件
        jsonl_path = os.path.join(root, "annotations.jsonl")
        self.jsonl_file = open(jsonl_path, "w", encoding="utf-8")
    
    # def save(self, root, data, idx):
    #     image = data["image"]
    #     label = data["label"]
    #     quality = data["quality"]
    #     mask = data["mask"]
    #     bboxes = data["bboxes"]
    #     font_info = data["font_info"]  # 获取字体信息
    #     transform_type = data["transform_type"]  # 获取变换类型

    #     # 转换图像数据格式
    #     image = Image.fromarray(image[..., :3].astype(np.uint8))
    #     mask = Image.fromarray(mask.astype(np.uint8))

    #     # 处理边界框坐标
    #     coords = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
    #     coords = "\t".join([",".join(map(str, map(int, coord))) for coord in coords])

    #     # 定义文件路径
    #     shard = str(idx // 10000)
    #     image_key = os.path.join("images", shard, f"{idx}.jpg")
    #     mask_key = os.path.join("masks", shard, f"{idx}.png")
    #     image_path = os.path.join(root, image_key)
    #     mask_path = os.path.join(root, mask_key)

    #     # 保存图像文件
    #     os.makedirs(os.path.dirname(image_path), exist_ok=True)
    #     image.save(image_path, quality=quality)
        
    #     # 保存掩码文件（如果需要）
    #     if self.mask_output:
    #         os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    #         mask.save(mask_path)

    #     # 写入标注信息
    #     # self.gt_file.write(f"{image_key}\t{label}\n")
    #     self.gt_file.write(f"{image_key}\t{label}\t{font_info}\t{transform_type}\n")
        
    #     # 写入坐标信息（如果需要）
    #     if self.coord_output and hasattr(self, 'coords_file'):
    #         self.coords_file.write(f"{image_key}\t{coords}\n")
    def save(self, root, data, idx):
        image = data["image"]
        labels = data["labels"]
        quality = data["quality"]
        mask = data["mask"]
        bboxes = data["bboxes"]
        font_infos = data["font_infos"]
        transforms_type = data["transforms_type"]

        # 转换图像数据格式
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

        # 定义文件路径
        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        mask_key = os.path.join("masks", shard, f"{idx}.png")
        image_path = os.path.join(root, image_key)
        mask_path = os.path.join(root, mask_key)

        # 保存图像文件
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path, quality=quality)
        
        # 保存掩码文件（如果需要）
        if self.mask_output:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            mask.save(mask_path)

        # 处理边界框坐标，转换为[x,y,x+w,y+h]格式
        processed_bboxes = []
        for x, y, w, h in bboxes:
            processed_bboxes.append([int(x), int(y), int(x + w), int(y + h)])
        
        # 处理文本标注，这里假设一个图像对应一个文本和一个边界框
        # 如果需要支持多个文本和边界框，可以根据实际情况调整
        # annotations = [label]  # 如果有多段文本，可以修改这里

        # 构建JSON对象
        json_obj = {
            'id' : image_key,
            "image_path": image_path , #改成image 绝对路径
            "ori_bboxs": processed_bboxes,
            "annotations": labels,
            # 可以根据需要添加额外的字段
            "font_infos": font_infos,
            "transform_types": transforms_type
        }
        
        # 写入JSONL文件
        self.jsonl_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    # def end_save(self, root):
    #     self.gt_file.close()
    #     if self.coord_output:
    #         self.coords_file.close()
    #     if self.glyph_coord_output:
    #         self.glyph_coords_file.close()
    def end_save(self, root):
        # 只关闭一个jsonl文件
        self.jsonl_file.close()

    def _generate_color(self):
        mg_color = self.color.sample()
        fg_style = self.style.sample()
        mg_style = self.style.sample()

        if fg_style["state"]:
            fg_color, bg_color, style_color = self.colormap3.sample()
            fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
        else:
            fg_color, bg_color = self.colormap2.sample()

        return fg_color, fg_style, mg_color, mg_style, bg_color
    def extract_info(self,font,transform):
        font_info = os.path.basename(font['path']).replace(".ttf", f"_{font['size']}")
        transform_type = "no_transform"
        # 解析Switch和Selector结构，提取变换类型
        if isinstance(transform, dict) and "state" in transform:
            if transform["state"]:
                selector_meta = transform["meta"]
                if isinstance(selector_meta, dict) and "idx" in selector_meta:
                    # 从template_zn_debug.py中的定义，变换组件列表是:
                    transform_types = ["Perspective", "Perspective", "Trapezoidate", "Trapezoidate", "Skew", "Skew", "Rotate"]
                    # 使用idx映射到实际的变换类型
                    transform_idx = selector_meta["idx"]
                    if 0 <= transform_idx < len(transform_types):
                        transform_type = transform_types[transform_idx]
        return font_info, transform_type
    def _generate_multiple_text(self, size, bg_image):
        # 随机确定要生成的元素数量
        target_num_elements = np.random.randint(self.element_nums[0], self.element_nums[1] + 1)
        
        # 初始化存储有效元素信息的列表
        valid_elements = []
        valid_element_layers = []
        
        # 初始化布局状态
        layout_state = self.element_layout.init_layout(size)
        
        # 创建当前状态的背景图像副本，用于可见性检查和融合
        current_bg = bg_image.copy()
        
        # 最大重试次数
        max_retries = 5
        
        # 为了调试，保存中间结果（可选保留）
        debug = False
        max_total_retries = 100
        current_total_retries = 0
        # 逐个尝试添加元素
        while len(valid_elements) < target_num_elements or len(valid_elements)==0:#确保至少有一个元素
            current_total_retries += 1
            if current_total_retries > max_total_retries:  # 如果总体重试次数过多，就跳出循环
                break
            retry_stat=True
            retry_count = 0
            while retry_stat:
                if retry_count == max_retries-1:
                    retry_stat=False

                retry_count += 1
                
                # 生成一个元素
                fg_color, fg_style, _, _, _ = self._generate_color()
                element  = self._generate_text_v3(fg_color, fg_style)  # 渲染一个文本行 with zaozi
                # if not element_render_success:
                    # continue
                # 创建元素图层的副本，避免修改原始图层
                element_layer_copy = element["layer"].copy()
                
                # 尝试添加元素到布局
                if self.element_layout.try_add_element(element_layer_copy, layout_state):
                    # 创建仅包含新元素的临时图像用于可见性检查
                    temp_element_image = utils.create_image(size)
                    quad_relative = element_layer_copy.quad - [0, 0]
                    utils.paste_image(element_layer_copy.image, temp_element_image, quad_relative)
                      
                    # 检查新元素在当前背景中的可见性
                    # 这里使用current_bg进行可见性检查，这样可以考虑已经添加的元素
                    blend_modes = np.random.permutation(BLEND_MODES)

                    for blend_mode in blend_modes:
                        out = utils.blend_image(temp_element_image, current_bg.copy(), mode=blend_mode)
                        debug_vis = False
                        if debug_vis:
                            debug_dir = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/synthtiger/examples/synthtiger/debug'
                            os.makedirs(debug_dir, exist_ok=True)
                            # 保存每个尝试的融合结果
                            debug_path = os.path.join(debug_dir, f'debug_try_{blend_mode}.png')
                            # 使用PIL保存图像
                            Image.fromarray(out[..., :3].astype(np.uint8)).save(debug_path)
                            print(f"Debug: Blend mode {blend_mode} saved to {debug_path}")
                            # 保存掩码
                            mask_path = os.path.join(debug_dir, f'debug_try_{blend_mode}_mask.png')
                            mask = Image.fromarray(temp_element_image[..., 3].astype(np.uint8))
                            mask.save(mask_path)
                            print(f"Debug: Blend mode {blend_mode} mask saved to {mask_path}")
                        if not self.visibility_check or _check_visibility(out, temp_element_image[..., 3]):
                            # 如果可见性检查通过，则接受这个元素并立即融合到当前背景中
                            valid_elements.append(element)
                            valid_element_layers.append(element_layer_copy)
                            
                            # 实际将元素融合到当前背景中，使用_blend_images函数确保混合模式正确
                            current_bg = out
                            break
                        else:
                            # print(f"Debug: Blend mode {blend_mode} failed visibility check")
                            pass
                        
                    
                    
            
        # 创建字体图像用于提取掩码
        font_image = utils.create_image(size)
        for layer in valid_element_layers:
            quad_relative = layer.quad - [0, 0]
            utils.paste_image(layer.image, font_image, quad_relative)
        
        
        if debug:
            debug_dir = '/mnt/bn/ocr-doc-nas/zhuhanshen/project/TextEvaluator/hanzi_writer/synthtiger/examples/synthtiger/debug'
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time() * 1000)
            
            # 保存最终图像
            final_path = os.path.join(debug_dir, f'debug_final_{timestamp}.png')
            final_image = Image.fromarray(current_bg[..., :3].astype(np.uint8))
            final_image.save(final_path)
            print(f"Debug: Final image saved to {final_path}")
            # 保存最终掩码
            final_path = os.path.join(debug_dir, f'debug_final_mask_{timestamp}.png')
            final_image = Image.fromarray(font_image[..., 3].astype(np.uint8))
            final_image.save(final_path)
            print(f"Debug: Final image saved to {final_path}")
        
        # 获取每个元素的信息
        bboxes = [layer.bbox for layer in valid_element_layers]
        if debug:
            #将bbox可视化在final_image上
            final_image = Image.fromarray(current_bg[..., :3].astype(np.uint8))
            draw = ImageDraw.Draw(final_image)
            for bbox in bboxes:
                # 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
                x, y, w, h = bbox
                draw.rectangle((x, y, x + w, y + h), outline="red", width=2)
            final_path = os.path.join(debug_dir, f'debug_final_bbox_{timestamp}.png')
            final_image.save(final_path)
            print(f"Debug: Final image saved to {final_path}")
        font_infos = [element["font_info"] for element in valid_elements]
        transform_types = [element["transform_type"] for element in valid_elements]
        labels = [element["label"] for element in valid_elements]
        
        return current_bg, font_image, bboxes, font_infos, transform_types, labels

    def _generate_text_v2(self, color, style):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        vertical = np.random.random() < self.vertical_prob 

        font = self.font.sample({"text": text, "vertical": vertical})
        font_size = font['size']

        char_layers = [layers.TextLayer(char, **font) for char in chars]
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical":  vertical }})
        # char_glyph_layers = [char_layer.copy() for char_layer in char_layers]

        text_layer = layers.Group(char_layers).merge()
        

        transform = self.transform.sample()
        self.color.apply([text_layer], color)
        self.texture.apply([text_layer])
        text_layer.ft_size = font_size
        self.style.apply([text_layer], style)
        self.transform.apply(
            [text_layer], transform
        )
        self.fit.apply([text_layer])
        self.pad.apply([text_layer])

        # for char_layer in char_layers:
        #     char_layer.topleft -= text_layer.topleft
        # for char_glyph_layer in char_glyph_layers:
        #     char_glyph_layer.topleft -= text_layer.topleft

        # out = text_layer.output()
        # bboxes = [char_layer.bbox for char_layer in char_layers]
        # bboxes = [text_layer.bbox]
        #extract font info & transform info
        font_info, transform_type = self.extract_info(font,transform)
        element = {
            "layer": text_layer,
            "label": label,
            "font_info": font_info,
            "transform_type": transform_type
        }
        return element         
    def _generate_text_v3(self, color, style):
        label = self.corpus.data(self.corpus.sample())

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)

        text = "".join(chars)
        vertical = np.random.random() < self.vertical_prob 

        font = self.font.sample({"text": text, "vertical": vertical})
        font_size = font['size']

        char_layers = [layers.ZaoziTextLayer(char=char, **font, **self.zaozi_config, pre_load_data=self.pre_load_data, hanzi_list=self.hanzi_list) for char in chars]
        final_labels = ''.join([char_layer.final_char for char_layer in char_layers])
        # element_render_success = all([char_layer.render_success for char_layer in char_layers])
        # if not element_render_success:
        #     return None, element_render_success
        self.shape.apply(char_layers)
        self.layout.apply(char_layers, {"meta": {"vertical":  vertical }})
        # char_glyph_layers = [char_layer.copy() for char_layer in char_layers]

        text_layer = layers.Group(char_layers).merge()
        

        transform = self.transform.sample()
        self.color.apply([text_layer], color)
        self.texture.apply([text_layer])
        text_layer.ft_size = font_size
        self.style.apply([text_layer], style)
        self.transform.apply(
            [text_layer], transform
        )
        self.fit.apply([text_layer])
        self.pad.apply([text_layer])

        # for char_layer in char_layers:
        #     char_layer.topleft -= text_layer.topleft
        # for char_glyph_layer in char_glyph_layers:
        #     char_glyph_layer.topleft -= text_layer.topleft

        # out = text_layer.output()
        # bboxes = [char_layer.bbox for char_layer in char_layers]
        # bboxes = [text_layer.bbox]
        #extract font info & transform info
        font_info, transform_type = self.extract_info(font,transform)
        element = {
            "layer": text_layer,
            "label": final_labels,
            "font_info": font_info,
            "transform_type": transform_type
        }
        # return element, element_render_success         
        return element

    def _generate_background(self, size, color):
        layer = layers.RectLayer(size)
        self.color.apply([layer], color)
        self.texture.apply([layer])
        out = layer.output()
        return out
    def _generate_background_v2(self, size):
        layer = layers.RectLayer(size)
        self.bg_texture.apply([layer])
        out = layer.output()
        return out

    # def _erase_image(self, image, mask):
    #     mask = _create_poly_mask(mask, self.foreground_mask_pad)
    #     mask_layer = layers.Layer(mask)
    #     image_layer = layers.Layer(image)
    #     image_layer.bbox = mask_layer.bbox
    #     self.midground_offset.apply([image_layer])
    #     out = image_layer.erase(mask_layer).output(bbox=mask_layer.bbox)
    #     return out

    def _postprocess_images(self, images):
        image_layers = [layers.Layer(image) for image in images]
        self.postprocess.apply(image_layers)
        outs = [image_layer.output() for image_layer in image_layers]
        return outs


def _blend_images(src, dst, visibility_check=False):
    blend_modes = np.random.permutation(BLEND_MODES)

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise RuntimeError("Text is not visible")

    return out


def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1


def _create_poly_mask(image, pad=0):
    height, width = image.shape[:2]
    alpha = image[..., 3].astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)

    cts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(cts, key=lambda ct: sum(cv2.boundingRect(ct)[:2]))

    if len(cts) == 1:
        hull = cv2.convexHull(cts[0])
        cv2.fillConvexPoly(mask, hull, 255)

    for idx in range(len(cts) - 1):
        pts = np.concatenate((cts[idx], cts[idx + 1]), axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    mask = utils.dilate_image(mask, pad)
    out = utils.create_image((width, height))
    out[..., 3] = mask
    return out