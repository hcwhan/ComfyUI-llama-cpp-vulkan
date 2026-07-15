"""BBox 工具链节点: JSON 解析转框, SEGS/MASK 转换, 索引选取."""

from collections import namedtuple

import numpy as np
import torch

from .....i18n.common_static import BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL, BBOX_MODE_SIMPLE, LOG_PREFIX
from .....i18n.common_static import CATEGORY as _CATEGORY
from .....i18n.lang import LANG
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

_BBOX = LANG["nodes"]["bbox"]
_LOGS = LANG["logs"]["bbox"]


def _normalized_label(value):
    """label 匹配归一化: 忽略大小写与首尾空格; None(字段缺失)视为不匹配.

    LLM 可能输出数字等非字符串标签, 与 bbox_label 的显示路径一致地强转 str,
    保证画得出的标签在过滤框中也能匹配到.
    """
    if value is None:
        return None
    return str(value).strip().casefold()


# 与 Impact Pack 的 SEG 保持同定义(modules/impact/core.py), 字段名与顺序不能改:
# 其部分节点依赖 namedtuple 语义(如 SEGSLabelAssign 调用 seg._replace)
SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


class json_to_bboxes:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (
                    [BBOX_MODE_SIMPLE, BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL],
                    {
                        "default": BBOX_MODE_SIMPLE,
                        "tooltip": _BBOX["json_to_bboxes"]["tooltips"]["mode"],
                    },
                ),
                "label": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": _BBOX["json_to_bboxes"]["tooltips"]["label"]},
                ),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    OUTPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")

    def process(self, json, mode, label, image=None):
        # INPUT_IS_LIST 下 widget 参数也会被包成列表
        mode = mode[0]
        wanted_label = _normalized_label(label[0])

        # image Instruct 逐张模式的 output 是分隔行拼接的整段文本,
        # 自动拆回逐张 JSON; 合法 JSON 文本中不存在真实分隔行, 不会被误拆
        json = [part for text in json for part in split_image_results(text)]

        # 拆平为 [1,H,W,C] 单帧列表, 记录每个输入元素的批次大小以便还原结构
        flat_images = []
        batch_sizes = []
        for img_item in image or []:
            img_batch = img_item.unsqueeze(0) if img_item.ndim == 3 else img_item
            batch_sizes.append(img_batch.shape[0])
            flat_images.extend(img_batch[n : n + 1] for n in range(img_batch.shape[0]))

        if mode in QWEN_BBOX_MODES and not flat_images:
            raise ValueError(_BBOX["json_to_bboxes"]["errors"]["image_required"])
        if flat_images and len(json) != len(flat_images):
            detail = _LOGS["detail_extra_json"] if len(json) > len(flat_images) else _LOGS["detail_extra_frames"]
            logger.warning(
                LOG_PREFIX + _LOGS["json_frame_mismatch"].format(json_count=len(json), frame_count=len(flat_images), detail=detail)
            )

        output_bboxes = []
        drawn_images = []

        for i, json_str in enumerate(json):
            try:
                items = parse_json(json_str)
            except ValueError as e:
                # 逐张模式拆出几十段结果时定位坏段, 与画框失败分支的 JSON #{i} 对齐
                raise ValueError(_BBOX["json_to_bboxes"]["errors"]["json_parse_failed"].format(i=i, error=e)) from None
            # 模型只检出单个目标时可能直接输出对象而非单元素列表
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                raise ValueError(_BBOX["json_to_bboxes"]["errors"]["not_a_list"].format(type_name=type(items).__name__))
            if wanted_label:
                # 兼容 label / text_content 混用的输出, 任一字段匹配即保留;
                # 非 dict 项原样保留, 由 json_to_pixel_bboxes 的结构校验给出
                # 带期望格式的报错, 而非在此抛裸 AttributeError
                items = [
                    b
                    for b in items
                    if not isinstance(b, dict)
                    or wanted_label in (_normalized_label(b.get("label")), _normalized_label(b.get("text_content")))
                ]

            if flat_images:
                curr_img = flat_images[min(i, len(flat_images) - 1)]
                _batch, height, width, _ch = curr_img.shape
                pixel_bboxes = json_to_pixel_bboxes(items, mode, width, height)
                try:
                    # draw_bbox 返回 [1,H,W,C]
                    drawn_images.append(draw_bbox(curr_img[0], pixel_bboxes, [bbox_label(b) for b in items]))
                except Exception as e:
                    logger.warning(LOG_PREFIX + _LOGS["draw_failed_json"].format(i=i, e=e))
                    # draw_bbox 经 numpy 往返恒输出 CPU 张量, 回退帧统一 .cpu(),
                    # 避免 --gpu-only 下与画框帧 torch.cat 混拼报 device mismatch
                    drawn_images.append(curr_img.cpu())
            else:
                pixel_bboxes = json_to_pixel_bboxes(items, mode)

            output_bboxes.append(pixel_bboxes)

        # JSON 少于帧时, 尾部未配对的帧原样进入输出(不画框), 保持批次结构完整
        # (透传帧同样统一 .cpu(), 理由同上; CPU 张量的 .cpu() 是零拷贝 no-op)
        if flat_images and len(drawn_images) < len(flat_images):
            drawn_images.extend(f.cpu() for f in flat_images[len(drawn_images) :])

        # 画框结果按输入图像的批次结构重新分组
        restructured_images = []
        cursor = 0
        for count in batch_sizes:
            restructured_images.append(torch.cat(drawn_images[cursor : cursor + count], dim=0))
            cursor += count
        # JSON 多于帧时, 多出的画框帧(复用末帧)作为单帧批次追加,
        # image_list 总帧数与 bboxes 组数保持对齐
        restructured_images.extend(drawn_images[cursor:])

        return (output_bboxes, restructured_images)


class bboxes_to_segs:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        # label/confidence 是检测结果本身的元数据(紧跟 bboxes 输入),
        # dilation/feather 作用于掩码矩形, crop_factor 决定上下文窗口, 按此语义分组排序
        tips = _BBOX["bboxes_to_segs"]["tooltips"]
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "label": (
                    "STRING",
                    {"default": "bbox", "tooltip": tips["label"]},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": tips["confidence"]},
                ),
                "dilation": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": tips["dilation"],
                    },
                ),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": tips["feather"]}),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": tips["crop_factor"],
                    },
                ),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)

    def process(self, bboxes, image, label, confidence, dilation, feather, crop_factor):
        batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)

        seg_list = []
        if batch_size > 1:
            logger.warning(LOG_PREFIX + _LOGS["segs_batch_first_frame"].format(batch_size=batch_size))
        image_for_cropping = image[0]

        for bbox in bboxes:
            coords = valid_int_bbox(bbox)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            # LLM 输出的坐标不可信, 先裁剪到图像范围
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                logger.warning(LOG_PREFIX + _LOGS["bbox_out_of_bounds"].format(bbox=bbox))
                continue

            # dilation 直接外扩掩码矩形(重绘区域), 与 bboxes_to_mask 及
            # Impact Pack 检测器的 dilation 语义一致; 限制在图像内,
            # 保证坐标不为负(Impact Pack 约定)
            mx1 = max(0, x1 - dilation)
            my1 = max(0, y1 - dilation)
            mx2 = min(width, x2 + dilation)
            my2 = min(height, y2 + dilation)

            # crop_region 以掩码矩形为中心按 crop_factor 放大(Impact Pack 惯例),
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
                # bbox 保留原始检测框(Impact Pack 约定 dilation 不改变 seg.bbox)
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                label=label,
            )

            seg_list.append(seg)

        segs = (mask_shape, seg_list)

        return (segs,)


class bboxes_to_mask:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        tips = _BBOX["bboxes_to_mask"]["tooltips"]
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": tips["dilation"],
                    },
                ),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": tips["feather"]}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        # 恒在 CPU 上构建与输出, 与 json_to_bboxes 的 image_list 策略一致:
        # --gpu-only 下跟随 image.device 会让直接 .numpy() 的下游第三方节点报错
        combined_full_mask = torch.zeros(mask_shape, dtype=torch.float32)
        # gaussian_filter 默认 truncate=4.0, 窗口向外留 4 sigma 即可覆盖全部有效衰减,
        # 在局部窗口内做羽化, 避免每个 bbox 都在全图尺寸上跑一次 filter
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
                logger.warning(LOG_PREFIX + _LOGS["bbox_empty_area"].format(bbox=bbox))
                continue

            # 局部窗口(含羽化边界), 裁剪到图像范围
            wx1, wy1 = max(0, x1_exp - margin), max(0, y1_exp - margin)
            wx2, wy2 = min(width, x2_exp + margin), min(height, y2_exp + margin)
            if wx2 <= wx1 or wy2 <= wy1:
                logger.warning(LOG_PREFIX + _LOGS["bbox_out_of_bounds"].format(bbox=bbox))
                continue

            # 扩张框(裁剪到图像内)在窗口坐标系中的位置
            inner_rect = (
                max(0, x1_exp) - wx1,
                max(0, y1_exp) - wy1,
                min(width, x2_exp) - wx1,
                min(height, y2_exp) - wy1,
            )
            local_mask_np = feathered_rect_mask(wy2 - wy1, wx2 - wx1, inner_rect, feather)
            local_mask_tensor = torch.from_numpy(local_mask_np)
            region = combined_full_mask[wy1:wy2, wx1:wx2]
            combined_full_mask[wy1:wy2, wx1:wx2] = torch.maximum(region, local_mask_tensor)

        return (combined_full_mask.unsqueeze(0),)


class bboxes_to_bbox:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    # 上游 json_to_bboxes 的 BBOX 输出是 OUTPUT_IS_LIST(每元素一组).
    # 必须声明 INPUT_IS_LIST 才能在单次调用中拿到完整的组列表,
    # 否则 ComfyUI 按组 map 执行, image_index/bbox_index 的二级索引语义失效.
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "bbox_index": (
                    "INT",
                    {"default": 0, "min": -998, "max": 999, "step": 1, "tooltip": _BBOX["bboxes_to_bbox"]["tooltips"]["bbox_index"]},
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)

    def process(self, bboxes, image_index, bbox_index):
        # INPUT_IS_LIST 下 widget 参数也会被包成列表
        image_index = image_index[0]
        bbox_index = bbox_index[0]
        if not 0 <= image_index < len(bboxes):
            raise IndexError(
                _BBOX["bboxes_to_bbox"]["errors"]["image_index_out_of_range"].format(image_index=image_index, count=len(bboxes))
            )
        group = bboxes[image_index]
        if bbox_index == 999:
            return (group,)
        if not -len(group) <= bbox_index < len(group):
            raise IndexError(
                _BBOX["bboxes_to_bbox"]["errors"]["bbox_index_out_of_range"].format(
                    bbox_index=bbox_index, image_index=image_index, count=len(group)
                )
            )
        return ([group[bbox_index]],)
