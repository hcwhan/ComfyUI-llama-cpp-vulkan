"""BBox 节点的强相关工具: 坐标换算, 画框, 羽化 mask, 结构校验."""

import hashlib

import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from ..encoding import tensor_to_uint8

QWEN_BBOX_MODES = ("Qwen3-VL", "Qwen2.5-VL")


def bbox_label(item):
    """取 bbox JSON 项的标签,兼容 label / text_content 两种字段。"""
    return item.get("label") or item.get("text_content") or "bbox"


def json_to_pixel_bboxes(json_items, mode, width=0, height=0):
    """把 LLM 输出的 bbox JSON 项换算为像素坐标 [(x0, y0, x1, y1), ...]。

    Qwen 系列模型输出 0-1000 归一化坐标,需按图像尺寸换算;
    simple 模式视为已是像素坐标,原样透传。
    """
    bboxes = []
    for item in json_items:
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in QWEN_BBOX_MODES:
            x0 = x0 / 1000 * width
            y0 = y0 / 1000 * height
            x1 = x1 / 1000 * width
            y1 = y1 / 1000 * height
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def _label_color(label):
    # 由 label 内容哈希出稳定颜色,同一 label 每次运行颜色一致;
    # 80-180 区间保证中等亮度,白色标签文字可读
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return tuple(80 + b % 101 for b in digest[:3])


def draw_bbox(image, pixel_bboxes, labels):
    img = Image.fromarray(tensor_to_uint8(image))
    draw = ImageDraw.Draw(img)

    for (x0, y0, x1, y1), label in zip(pixel_bboxes, labels):
        color = _label_color(label)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)


def valid_int_bbox(bbox):
    """校验 bbox 结构并取整为 (x1, y1, x2, y2),非法时打 warning 返回 None。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        print(f"Warning: Skipping invalid bbox item: {bbox}")
        return None
    return tuple(int(v) for v in bbox[:4])


def feathered_rect_mask(window_h, window_w, inner_rect, feather):
    """在 (window_h, window_w) 局部窗口内构建矩形 mask,feather > 0 时做高斯羽化。

    inner_rect 是窗口坐标系下的 (x1, y1, x2, y2)。
    在局部窗口而非全图上跑 gaussian_filter,避免每个 bbox 的羽化代价随图像尺寸增长。
    """
    mask = np.zeros((window_h, window_w), dtype=np.float32)
    x1, y1, x2, y2 = inner_rect
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    if feather > 0:
        mask = gaussian_filter(mask, sigma=feather)
    return mask
