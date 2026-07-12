"""BBox 节点的强相关工具: 坐标换算, 画框, 羽化 mask, 结构校验."""

import hashlib
from functools import lru_cache

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from .....shared.logger import logger
from ..encoding import tensor_to_uint8

QWEN_BBOX_MODES = ("Qwen3-VL", "Qwen2.5-VL")

# label 常为中文(BBox 检测预设引导用户填中文类别),PIL 默认 bitmap 字体
# 无 CJK 字形会画成占位方块,按平台常见 CJK 字体依次尝试
_CJK_FONT_CANDIDATES = (
    "msyh.ttc",                 # Windows 微软雅黑
    "simhei.ttf",               # Windows 黑体
    "NotoSansCJK-Regular.ttc",  # Linux Noto CJK
    "NotoSansSC-Regular.otf",
    "wqy-zenhei.ttc",           # Linux 文泉驿正黑
)


@lru_cache(maxsize=8)
def _label_font(size):
    """按字号加载 CJK 字体,全部候选缺失时回退 PIL 默认字体(中文会显示为方块)。"""
    for name in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    logger.warning("[llama-cpp-vulkan] No CJK font found, bbox labels may render as boxes")
    return ImageFont.load_default(size)


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
        # LLM 输出结构不可信,显式校验并给出期望格式,避免裸 KeyError/TypeError
        if not isinstance(item, dict):
            raise ValueError(f'Expected a JSON list of objects like {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}, got item: {item!r}')
        coords = item.get("bbox_2d")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f'BBox item is missing a valid "bbox_2d": [x1, y1, x2, y2] field: {item!r}')
        x0, y0, x1, y1 = coords
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

    # 字号/线宽随图像尺寸缩放,高分辨率图上固定像素值会小到不可读
    ref = min(img.size)
    font_size = max(12, ref // 40)
    line_width = max(2, ref // 250)
    font = _label_font(font_size)

    for (x0, y0, x1, y1), label in zip(pixel_bboxes, labels):
        color = _label_color(label)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        text_y = max(0, y0 - font_size - 4)
        text_size = draw.textbbox((x0, text_y), label, font=font)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255), font=font)
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)


def valid_int_bbox(bbox):
    """校验 bbox 结构并取整为 (x1, y1, x2, y2),非法时打 warning 返回 None。

    Qwen 归一化坐标换算后是浮点数,四舍五入比截断更贴近原框;
    坐标值来自 LLM 输出,非数字时按无效项跳过。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        logger.warning(f"[llama-cpp-vulkan] Skipping invalid bbox item: {bbox}")
        return None
    try:
        return tuple(int(round(float(v))) for v in bbox[:4])
    except (TypeError, ValueError):
        logger.warning(f"[llama-cpp-vulkan] Skipping bbox with non-numeric coordinates: {bbox}")
        return None


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
