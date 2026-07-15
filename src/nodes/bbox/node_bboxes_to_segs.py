"""BBoxes to SEGS 节点, 把 BBox 转换为 Impact Pack 兼容的 SEGS 格式, 供 Detailer 等节点做局部重绘."""

from collections import namedtuple

import numpy as np

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.common_static import LOG_PREFIX
from ...i18n.lang import LANG
from ...shared.logger import logger
from .bbox_utils import feathered_rect_mask, valid_int_bbox

_TIPS = LANG["nodes"]["bbox"]["bboxes_to_segs"]["tooltips"]
_LOGS = LANG["logs"]["bbox"]

# 与 Impact Pack 的 SEG 保持同定义(modules/impact/core.py), 字段名与顺序不能改:
# 其部分节点依赖 namedtuple 语义(如 SEGSLabelAssign 调用 seg._replace)
SEG = namedtuple(
    "SEG",
    ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"],
    defaults=[None],
)


class bboxes_to_segs:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        # label/confidence 是检测结果本身的元数据(紧跟 bboxes 输入),
        # dilation/feather 作用于掩码矩形, crop_factor 决定上下文窗口, 按此语义分组排序
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "label": (
                    "STRING",
                    {"default": "bbox", "tooltip": _TIPS["label"]},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TIPS["confidence"]},
                ),
                "dilation": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": _TIPS["dilation"],
                    },
                ),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": _TIPS["feather"]}),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": _TIPS["crop_factor"],
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
            # Impact Pack 的 SEG 约定 cropped_image 为 [1, H, W, C];
            # 恒在 CPU 上构建与输出, 与 bboxes_to_mask / json_to_bboxes 策略一致:
            # --gpu-only 下跟随 image.device 会让直接 .numpy() 的下游第三方节点报错
            cropped_image_tensor = image_for_cropping[cy1:cy2, cx1:cx2, :].unsqueeze(0).cpu()

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
