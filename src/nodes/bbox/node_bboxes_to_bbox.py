"""BBoxes to BBox 节点, 按 image_index/bbox_index 二级索引从 BBox 组列表中选取."""

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix

_TIPS = LANG["nodes"]["bbox"]["bboxes_to_bbox"]["tooltips"]
_ERRORS = LANG["nodes"]["bbox"]["bboxes_to_bbox"]["errors"]
_LOGS = LANG["logs"]["bbox"]
_PREFIX = node_log_prefix("BBoxes to BBox")


class bboxes_to_bbox:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    # 上游 json_to_bboxes 的 BBOX 输出是 OUTPUT_IS_LIST(每元素一组).
    # 必须声明 INPUT_IS_LIST 才能在单次调用中拿到完整的组列表,
    # 否则 ComfyUI 按组 map 执行, image_index/bbox_index 的二级索引语义失效.
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": _TIPS["image_index"]},
                ),
                "bbox_index": (
                    "INT",
                    {"default": 0, "min": -998, "max": 999, "step": 1, "tooltip": _TIPS["bbox_index"]},
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
            raise IndexError(_ERRORS["image_index_out_of_range"].format(image_index=image_index, count=len(bboxes)))
        group = bboxes[image_index]
        if bbox_index == 999:
            logger.info(_PREFIX + _LOGS["bbox_selected_all"].format(image_index=image_index, count=len(group)))
            return (group,)
        if not -len(group) <= bbox_index < len(group):
            raise IndexError(_ERRORS["bbox_index_out_of_range"].format(bbox_index=bbox_index, image_index=image_index, count=len(group)))
        selected = group[bbox_index]
        logger.info(_PREFIX + _LOGS["bbox_selected"].format(image_index=image_index, bbox_index=bbox_index, bbox=selected))
        return ([selected],)
