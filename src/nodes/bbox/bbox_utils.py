"""BBox 节点的强相关工具: 坐标换算, 画框, 羽化 mask, 结构校验."""

import hashlib
import math
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from ...i18n.common_static import BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL, LOG_PREFIX
from ...i18n.lang import LANG
from ...shared.encoding import tensor_to_uint8
from ...shared.logger import logger

_ERRORS = LANG["nodes"]["bbox"]["json_to_bboxes"]["errors"]
_LOGS = LANG["logs"]["bbox"]

QWEN_BBOX_MODES = (BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL)

# Qwen2.5-VL 输出的是 mtmd smart_resize 后图像空间的绝对像素坐标(官方语义,
# 实测 Qwen2.5-VL-3B GGUF 逐值吻合), 换算回原图需要复现 resize 尺寸.
# 三个常量对接 requirements.txt 固定 wheel 的 mtmd 默认值(经 5 组不同尺寸
# 实测反推并完全复现): 有效 patch 28 (patch 14 x merge 2), 下限 8 token,
# 上限 4096 token.
# 注意 loader 的 image_min/max_tokens > 0 会改变上下限, 此时换算不再精确.
_QWEN25_FACTOR = 28
_QWEN25_MIN_PIXELS = 8 * 28 * 28
_QWEN25_MAX_PIXELS = 4096 * 28 * 28


# floor/ceil 前的浮点容差: beta 缩放的精确值可能恰为整数(如 2400x2400 时
# 2400/beta 精确等于 1792), 浮点误差会让 floor 少算一个 patch
_EPS = 1e-6


def qwen25_smart_resize(width, height):
    """复现 Qwen2.5-VL 预处理的 smart_resize, 返回 mtmd 实际送入模型的 (宽, 高).

    与官方参考实现一致: 宽高先四舍五入到 28 倍数; 面积超上限时按 beta 缩小后
    向下取整到 28 倍数; 低于下限时放大后向上取整到 28 倍数.
    """
    f = _QWEN25_FACTOR
    w_bar = max(f, round(width / f) * f)
    h_bar = max(f, round(height / f) * f)
    if w_bar * h_bar > _QWEN25_MAX_PIXELS:
        beta = math.sqrt(width * height / _QWEN25_MAX_PIXELS)
        w_bar = max(f, math.floor(width / beta / f + _EPS) * f)
        h_bar = max(f, math.floor(height / beta / f + _EPS) * f)
    elif w_bar * h_bar < _QWEN25_MIN_PIXELS:
        beta = math.sqrt(_QWEN25_MIN_PIXELS / (width * height))
        w_bar = math.ceil(width * beta / f - _EPS) * f
        h_bar = math.ceil(height * beta / f - _EPS) * f
    return w_bar, h_bar


# label 常为中文(BBox 检测预设引导用户填中文类别), PIL 默认 bitmap 字体
# 无 CJK 字形会画成占位方块, 按平台常见 CJK 字体依次尝试
_CJK_FONT_CANDIDATES = (
    "msyh.ttc",  # Windows 微软雅黑
    "simhei.ttf",  # Windows 黑体
    "NotoSansCJK-Regular.ttc",  # Linux Noto CJK
    "NotoSansSC-Regular.otf",
    "wqy-zenhei.ttc",  # Linux 文泉驿正黑
)


@lru_cache(maxsize=8)
def _label_font(size):
    """按字号加载 CJK 字体, 全部候选缺失时回退 PIL 默认字体(中文会显示为方块)."""
    for name in _CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    logger.warning(LOG_PREFIX + _LOGS["no_cjk_font"])
    return ImageFont.load_default(size)


def bbox_label(item):
    """取 bbox JSON 项的标签, 兼容 label / text_content 两种字段.

    LLM 可能输出数字等非字符串标签, 强转 str 保证下游
    _label_color(label.encode) 与 PIL draw.text 不因单个标签放弃整张图.
    """
    return str(item.get("label") or item.get("text_content") or "bbox")


def json_to_pixel_bboxes(json_items, mode, width=0, height=0):
    """把 LLM 输出的 bbox JSON 项换算为原图像素坐标 [(x0, y0, x1, y1), ...].

    - Qwen3-VL:   输出 0-1000 归一化坐标, 按原图尺寸换算
    - Qwen2.5-VL: 输出 smart_resize 后图像空间的绝对坐标, 按 原图/resize 比例还原
    - simple:     视为已是原图像素坐标, 原样透传
    """
    if mode == BBOX_MODE_QWEN25_VL:
        rw, rh = qwen25_smart_resize(width, height)
        sx, sy = width / rw, height / rh
    elif mode == BBOX_MODE_QWEN3:
        sx, sy = width / 1000, height / 1000
    else:
        sx = sy = 1.0

    bboxes = []
    for item in json_items:
        # LLM 输出结构不可信, 显式校验并给出期望格式, 避免裸 KeyError/TypeError
        if not isinstance(item, dict):
            raise ValueError(_ERRORS["item_not_object"].format(item=item))
        coords = item.get("bbox_2d")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(_ERRORS["missing_bbox_2d"].format(item=item))
        try:
            # 坐标经 float 强转: 弱模型常见输出数字字符串, 与 valid_int_bbox 行为一致
            x0, y0, x1, y1 = (float(v) for v in coords)
        except (TypeError, ValueError):
            raise ValueError(_ERRORS["coords_not_numeric"].format(item=item)) from None
        bboxes.append((x0 * sx, y0 * sy, x1 * sx, y1 * sy))
    return bboxes


def _label_color(label):
    # 由 label 内容哈希出稳定颜色, 同一 label 每次运行颜色一致;
    # 80-180 区间保证中等亮度, 白色标签文字可读
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return tuple(80 + b % 101 for b in digest[:3])


def draw_bbox(image, pixel_bboxes, labels):
    img = Image.fromarray(tensor_to_uint8(image))
    draw = ImageDraw.Draw(img)

    # 字号/线宽随图像尺寸缩放, 高分辨率图上固定像素值会小到不可读
    ref = min(img.size)
    font_size = max(12, ref // 40)
    line_width = max(2, ref // 250)
    font = _label_font(font_size)

    # 两者均由调用方从同一 items 列表逐项生成, 长度恒等, strict 仅作断言
    for (x0, y0, x1, y1), label in zip(pixel_bboxes, labels, strict=True):
        color = _label_color(label)
        try:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
            text_y = max(0, y0 - font_size - 4)
            text_size = draw.textbbox((x0, text_y), label, font=font)
            draw.rectangle([text_size[0], text_size[1] - 2, text_size[2] + 4, text_size[3] + 2], fill=color)
            draw.text((x0 + 2, text_y), label, fill=(255, 255, 255), font=font)
        except Exception as e:
            # 反向坐标(x1 < x0, LLM 常见错误)或非有限值会让 PIL 抛错;
            # 逐框跳过, 与 SEGS/MASK 路径的逐框容错粒度一致,
            # 单个坏框不放弃整张图的其余框
            logger.warning(LOG_PREFIX + _LOGS["bbox_draw_failed"].format(label=label, x0=x0, y0=y0, x1=x1, y1=y1, e=e))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)


def valid_int_bbox(bbox):
    """校验 bbox 结构并取整为 (x1, y1, x2, y2), 非法时打 warning 返回 None.

    Qwen 归一化坐标换算后是浮点数, 四舍五入比截断更贴近原框;
    坐标值来自 LLM 输出, 非数字时按无效项跳过.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        logger.warning(LOG_PREFIX + _LOGS["bbox_invalid_item"].format(bbox=bbox))
        return None
    try:
        return tuple(int(round(float(v))) for v in bbox[:4])
    except (TypeError, ValueError):
        logger.warning(LOG_PREFIX + _LOGS["bbox_non_numeric"].format(bbox=bbox))
        return None


def feathered_rect_mask(window_h, window_w, inner_rect, feather):
    """在 (window_h, window_w) 局部窗口内构建矩形 mask, feather > 0 时做高斯羽化.

    inner_rect 是窗口坐标系下的 (x1, y1, x2, y2).
    在局部窗口而非全图上跑 gaussian_filter, 避免每个 bbox 的羽化代价随图像尺寸增长.
    """
    mask = np.zeros((window_h, window_w), dtype=np.float32)
    x1, y1, x2, y2 = inner_rect
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    if feather > 0:
        mask = gaussian_filter(mask, sigma=feather)
    return mask
