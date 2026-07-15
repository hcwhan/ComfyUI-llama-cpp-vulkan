"""BBoxes to MASK 节点, 把 BBox 合成为单张 MASK, 供 Inpaint / 遮罩合成等下游使用."""

import torch

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.common_static import LOG_PREFIX
from ...i18n.lang import LANG
from ...shared.logger import logger
from .bbox_utils import feathered_rect_mask, valid_int_bbox

_TIPS = LANG["nodes"]["bbox"]["bboxes_to_mask"]["tooltips"]
_LOGS = LANG["logs"]["bbox"]


class bboxes_to_mask:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
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
