"""BBoxes to MASK 节点, 把 BBox 合成为单张 MASK, 供 Inpaint / 遮罩合成等下游使用."""

import torch

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix
from .bbox_utils import feathered_rect_mask, valid_int_bbox

_TIPS = LANG["nodes"]["bbox"]["bboxes_to_mask"]["tooltips"]
_LOGS = LANG["logs"]["bbox"]
_PREFIX = node_log_prefix("BBoxes to MASK")


class bboxes_to_mask:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
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
                        "tooltip": _TIPS["dilation"],
                    },
                ),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": _TIPS["feather"]}),
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
            coords = valid_int_bbox(bbox, log_prefix=_PREFIX)
            if coords is None:
                continue
            x1, y1, x2, y2 = coords
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation

            if x2_exp - x1_exp <= 0 or y2_exp - y1_exp <= 0:
                logger.warning(_PREFIX + _LOGS["bbox_empty_area"].format(bbox=bbox))
                continue

            # 出界判定作用在不含 margin 的扩张框上, 与 bboxes_to_segs 口径一致:
            # 修复前判定作用在扩了 margin 的窗口上, feather > 0 且扩张框完全
            # 出界但距图缘不足 margin 时窗口被撑成非空, 无效框静默跳过且白跑
            # 一次高斯滤波
            mx1, my1 = max(0, x1_exp), max(0, y1_exp)
            mx2, my2 = min(width, x2_exp), min(height, y2_exp)
            if mx2 <= mx1 or my2 <= my1:
                logger.warning(_PREFIX + _LOGS["bbox_out_of_bounds"].format(bbox=bbox))
                continue

            # 局部窗口(含羽化边界), 裁剪到图像范围;
            # margin 只用于撑羽化窗口, 扩张框非空时窗口必非空
            wx1, wy1 = max(0, x1_exp - margin), max(0, y1_exp - margin)
            wx2, wy2 = min(width, x2_exp + margin), min(height, y2_exp + margin)

            # 扩张框(裁剪到图像内)在窗口坐标系中的位置
            inner_rect = (mx1 - wx1, my1 - wy1, mx2 - wx1, my2 - wy1)
            local_mask_np = feathered_rect_mask(wy2 - wy1, wx2 - wx1, inner_rect, feather)
            local_mask_tensor = torch.from_numpy(local_mask_np)
            region = combined_full_mask[wy1:wy2, wx1:wx2]
            combined_full_mask[wy1:wy2, wx1:wx2] = torch.maximum(region, local_mask_tensor)

        logger.info(
            _PREFIX
            + _LOGS["mask_summary"].format(
                bbox_count=len(bboxes),
                coverage=combined_full_mask.gt(0.0).float().mean().item() * 100,
                dilation=dilation,
                feather=feather,
            )
        )
        return (combined_full_mask.unsqueeze(0),)
