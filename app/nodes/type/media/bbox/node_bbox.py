"""BBox 工具链节点: JSON 解析转框, SEGS/MASK 转换, 索引选取."""

from collections import namedtuple

import torch
import numpy as np

from .....shared.logger import logger
from .....shared.text_utils import parse_json, split_image_results
from .bbox_utils import (
    QWEN_BBOX_MODES,
    bbox_label,
    draw_bbox,
    feathered_rect_mask,
    json_to_pixel_bboxes,
    valid_int_bbox,
)

def _normalized_label(value):
    """label 匹配归一化: 忽略大小写与首尾空格,非字符串视为不匹配。"""
    return value.strip().casefold() if isinstance(value, str) else None


# 与 Impact Pack 的 SEG 保持同定义(modules/impact/core.py),字段名与顺序不能改:
# 其部分节点依赖 namedtuple 语义(如 SEGSLabelAssign 调用 seg._replace)
SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


class json_to_bboxes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {
                    "default": "simple",
                    "tooltip": "坐标系换算:\nsimple = 原样透传 (模型输出即原图像素坐标)\nQwen3-VL = 0-1000 归一化坐标\nQwen2.5-VL = 内部 resize 空间的绝对坐标 (自动还原到原图;\n  loader 修改过 image_min/max_tokens 时换算会有偏差;\n  需配合 image Instruct 逐张模式使用, 批量模式的 max_size 缩放会破坏换算)"
                }),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "只保留指定 label 的 BBox.\n(匹配忽略大小写与首尾空格)"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, json, mode, label, image=None):
        # INPUT_IS_LIST 下 widget 参数也会被包成列表
        mode = mode[0]
        wanted_label = _normalized_label(label[0])

        # image Instruct 逐张模式的 output 是分隔行拼接的整段文本,
        # 自动拆回逐张 JSON;合法 JSON 文本中不存在真实分隔行,不会被误拆
        json = [part for text in json for part in split_image_results(text)]

        # 拆平为 [1,H,W,C] 单帧列表,记录每个输入元素的批次大小以便还原结构
        flat_images = []
        batch_sizes = []
        for img_batch in image or []:
            if img_batch.ndim == 3:
                img_batch = img_batch.unsqueeze(0)
            batch_sizes.append(img_batch.shape[0])
            flat_images.extend(img_batch[n:n + 1] for n in range(img_batch.shape[0]))

        if mode in QWEN_BBOX_MODES and not flat_images:
            raise ValueError("Image required for Qwen mode")
        if flat_images and len(json) != len(flat_images):
            if len(json) > len(flat_images):
                detail = "extra JSON entries reuse the last frame, appended to image_list as single-frame batches"
            else:
                detail = "unpaired trailing frames are passed through without boxes"
            logger.warning(f"[llama-cpp-vulkan] {len(json)} JSON result(s) but {len(flat_images)} image frame(s); pairing by index, {detail}")

        output_bboxes = []
        drawn_images = []

        for i, json_str in enumerate(json):
            items = parse_json(json_str)
            # 模型只检出单个目标时可能直接输出对象而非单元素列表
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                raise ValueError(f'Expected a JSON list of {{"bbox_2d": [...], "label": "..."}} objects, got: {type(items).__name__}')
            if wanted_label:
                # 兼容 label / text_content 混用的输出,任一字段匹配即保留
                items = [b for b in items if wanted_label in (_normalized_label(b.get("label")), _normalized_label(b.get("text_content")))]

            if flat_images:
                curr_img = flat_images[min(i, len(flat_images) - 1)]
                _batch, height, width, _ch = curr_img.shape
                pixel_bboxes = json_to_pixel_bboxes(items, mode, width, height)
                try:
                    # draw_bbox 返回 [1,H,W,C]
                    drawn_images.append(draw_bbox(curr_img[0], pixel_bboxes, [bbox_label(b) for b in items]))
                except Exception as e:
                    logger.warning(f"[llama-cpp-vulkan] Error drawing bboxes for JSON #{i}: {e}")
                    drawn_images.append(curr_img)
            else:
                pixel_bboxes = json_to_pixel_bboxes(items, mode)

            output_bboxes.append(pixel_bboxes)

        # JSON 少于帧时, 尾部未配对的帧原样进入输出(不画框), 保持批次结构完整
        if flat_images and len(drawn_images) < len(flat_images):
            drawn_images.extend(flat_images[len(drawn_images):])

        # 画框结果按输入图像的批次结构重新分组
        restructured_images = []
        cursor = 0
        for count in batch_sizes:
            restructured_images.append(torch.cat(drawn_images[cursor:cursor + count], dim=0))
            cursor += count
        # JSON 多于帧时, 多出的画框帧(复用末帧)作为单帧批次追加,
        # image_list 总帧数与 bboxes 组数保持对齐
        restructured_images.extend(drawn_images[cursor:])

        return (output_bboxes, restructured_images)


class bboxes_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        # label/confidence 是检测结果本身的元数据(紧跟 bboxes 输入),
        # dilation/feather 作用于掩码矩形, crop_factor 决定上下文窗口, 按此语义分组排序
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "label": ("STRING", {"default": "bbox", "tooltip": "写入每个 SEG 的 label, 供下游按 label 过滤/赋值 (如 Impact Pack 的 SEGS Filter)."}),
                "confidence": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "写入每个 SEG 的置信度, 供下游按阈值过滤."}),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": "掩码矩形向外扩张的像素数, 直接扩大下游的重绘区域.\n(与 Impact Pack 检测器及 BBoxes to MASK 的 dilation 语义一致)"}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "掩码边缘高斯羽化的 sigma (像素)."}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "crop_region 相对掩码矩形的放大倍数, 为下游 Detailer 提供重绘上下文.\n(Impact Pack 惯例, 1.0 = 不外扩)"}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, bboxes, image, label, confidence, dilation, feather, crop_factor):
        batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)

        seg_list = []
        if batch_size > 1:
            logger.warning(f"[llama-cpp-vulkan] BBoxes to SEGS received a batch of {batch_size} images; cropped images are taken from the first frame only")
        image_for_cropping = image[0]

        for bbox in bboxes:
            coords = valid_int_bbox(bbox)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            # LLM 输出的坐标不可信，先裁剪到图像范围
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"[llama-cpp-vulkan] Skipping bbox outside image bounds: {bbox}")
                continue

            # dilation 直接外扩掩码矩形（重绘区域），与 bboxes_to_mask 及
            # Impact Pack 检测器的 dilation 语义一致；限制在图像内，
            # 保证坐标不为负（Impact Pack 约定）
            mx1 = max(0, x1 - dilation)
            my1 = max(0, y1 - dilation)
            mx2 = min(width, x2 + dilation)
            my2 = min(height, y2 + dilation)

            # crop_region 以掩码矩形为中心按 crop_factor 放大（Impact Pack 惯例），
            # 供下游 Detailer 携带周边上下文重绘
            pad_x = int((mx2 - mx1) * (crop_factor - 1.0) / 2)
            pad_y = int((my2 - my1) * (crop_factor - 1.0) / 2)
            cx1 = max(0, mx1 - pad_x)
            cy1 = max(0, my1 - pad_y)
            cx2 = min(width, mx2 + pad_x)
            cy2 = min(height, my2 + pad_y)

            crop_region = [cx1, cy1, cx2, cy2]

            # 掩码矩形在 crop 窗口坐标系中的位置
            inner_rect = (mx1 - cx1, my1 - cy1, mx2 - cx1, my2 - cy1)
            cropped_mask_np = feathered_rect_mask(cy2 - cy1, cx2 - cx1, inner_rect, feather)
            # Impact Pack 的 SEG 约定 cropped_image 为 [1, H, W, C]
            cropped_image_tensor = image_for_cropping[cy1:cy2, cx1:cx2, :].unsqueeze(0)

            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                # Impact Pack 约定 confidence 为标量
                confidence=confidence,
                crop_region=crop_region,
                # bbox 保留原始检测框（Impact Pack 约定 dilation 不改变 seg.bbox）
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                label=label,
            )

            seg_list.append(seg)

        segs = (mask_shape, seg_list)

        return (segs,)


class bboxes_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1, "tooltip": "掩码矩形向外扩张的像素数.\n(与 BBoxes to SEGS 的 dilation 语义一致)"}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "掩码边缘高斯羽化的 sigma (像素)."}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32, device=image.device)
        # gaussian_filter 默认 truncate=4.0,窗口向外留 4 sigma 即可覆盖全部有效衰减,
        # 在局部窗口内做羽化,避免每个 bbox 都在全图尺寸上跑一次 filter
        margin = int(4 * feather) + 1 if feather > 0 else 0

        for bbox in bboxes:
            coords = valid_int_bbox(bbox)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation

            if x2_exp - x1_exp <= 0 or y2_exp - y1_exp <= 0:
                logger.warning(f"[llama-cpp-vulkan] Skipping bbox with empty area: {bbox}")
                continue

            # 局部窗口(含羽化边界),裁剪到图像范围
            wx1, wy1 = max(0, x1_exp - margin), max(0, y1_exp - margin)
            wx2, wy2 = min(width, x2_exp + margin), min(height, y2_exp + margin)
            if wx2 <= wx1 or wy2 <= wy1:
                logger.warning(f"[llama-cpp-vulkan] Skipping bbox outside image bounds: {bbox}")
                continue

            # 扩张框(裁剪到图像内)在窗口坐标系中的位置
            inner_rect = (
                max(0, x1_exp) - wx1, max(0, y1_exp) - wy1,
                min(width, x2_exp) - wx1, min(height, y2_exp) - wy1,
            )
            local_mask_np = feathered_rect_mask(wy2 - wy1, wx2 - wx1, inner_rect, feather)
            local_mask_tensor = torch.from_numpy(local_mask_np).to(image.device)
            region = combined_full_mask[wy1:wy2, wx1:wx2]
            combined_full_mask[wy1:wy2, wx1:wx2] = torch.maximum(region, local_mask_tensor)

        return (combined_full_mask.unsqueeze(0),)


class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": ("INT", {
                    "default": 0,
                    "min": -998,
                    "max": 999,
                    "step": 1,
                    "tooltip": "图内 BBox 索引. 设为 999 时返回该图全部 BBox."
                }),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    # 上游 json_to_bboxes 的 BBOX 输出是 OUTPUT_IS_LIST（每元素一组）。
    # 必须声明 INPUT_IS_LIST 才能在单次调用中拿到完整的组列表，
    # 否则 ComfyUI 按组 map 执行，image_index/bbox_index 的二级索引语义失效。
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, bboxes, image_index, bbox_index):
        # INPUT_IS_LIST 下 widget 参数也会被包成列表
        image_index = image_index[0]
        bbox_index = bbox_index[0]
        if not 0 <= image_index < len(bboxes):
            raise IndexError(f"image_index {image_index} out of range: only {len(bboxes)} bbox group(s) available")
        group = bboxes[image_index]
        if bbox_index == 999:
            return (group,)
        if not -len(group) <= bbox_index < len(group):
            raise IndexError(f"bbox_index {bbox_index} out of range: image {image_index} has {len(group)} bbox(es)")
        return ([group[bbox_index]],)
