"""JSON to BBoxes 节点, 把 LLM 输出的检测 JSON 转换为像素坐标 BBox, 可选输出画框预览图."""

import torch

from ...i18n.common_static import BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL, BBOX_MODE_SIMPLE
from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix
from ...shared.text_utils import parse_json, split_image_results
from .bbox_utils import QWEN_BBOX_MODES, bbox_label, draw_bbox, json_to_pixel_bboxes

_TIPS = LANG["nodes"]["bbox"]["json_to_bboxes"]["tooltips"]
_ERRORS = LANG["nodes"]["bbox"]["json_to_bboxes"]["errors"]
_LOGS = LANG["logs"]["bbox"]
_PREFIX = node_log_prefix("JSON to BBoxes")


def _normalized_label(value):
    """label 匹配归一化: 忽略大小写与首尾空格; None(字段缺失)不参与匹配.

    LLM 可能输出数字等非字符串标签, 与 bbox_label 的显示路径一致地强转 str,
    保证画得出的标签在过滤框中也能匹配到.
    (两字段均缺失的项由过滤分支经 bbox_label 取 fallback 标签 "bbox" 参与匹配)
    """
    if value is None:
        return None
    return str(value).strip().casefold()


class json_to_bboxes:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (
                    [BBOX_MODE_SIMPLE, BBOX_MODE_QWEN3, BBOX_MODE_QWEN25_VL],
                    {
                        "default": BBOX_MODE_SIMPLE,
                        "tooltip": _TIPS["mode"],
                    },
                ),
                "label": (
                    "STRING",
                    {"default": "", "multiline": False, "tooltip": _TIPS["label"]},
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

        # image Instruct 逐张模式的 output 是前缀行拼接的整段文本,
        # 自动拆回逐张 JSON; 合法 JSON 文本中不存在真实前缀行, 不会被误拆
        json = [part for text in json for part in split_image_results(text)]

        # 拆平为 [1,H,W,C] 单帧列表, 记录每个输入元素的批次大小以便还原结构
        flat_images = []
        batch_sizes = []
        for img_item in image or []:
            img_batch = img_item.unsqueeze(0) if img_item.ndim == 3 else img_item
            batch_sizes.append(img_batch.shape[0])
            flat_images.extend(img_batch[n : n + 1] for n in range(img_batch.shape[0]))

        if mode in QWEN_BBOX_MODES and not flat_images:
            raise ValueError(_ERRORS["image_required"])
        if flat_images and len(json) != len(flat_images):
            detail = _LOGS["detail_extra_json"] if len(json) > len(flat_images) else _LOGS["detail_extra_frames"]
            logger.warning(_PREFIX + _LOGS["json_frame_mismatch"].format(json_count=len(json), frame_count=len(flat_images), detail=detail))

        output_bboxes = []
        drawn_images = []

        for i, json_str in enumerate(json):
            try:
                items = parse_json(json_str)
            except ValueError as e:
                # 逐张模式拆出几十段结果时定位坏段; 序号从 1 起,
                # 与前缀行 "Image N" 及画框失败分支的 JSON #{i} 对齐
                raise ValueError(_ERRORS["json_parse_failed"].format(i=i + 1, error=e)) from None
            # 模型只检出单个目标时可能直接输出对象而非单元素列表
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                raise ValueError(_ERRORS["not_a_list"].format(type_name=type(items).__name__))
            if wanted_label:
                # 兼容 label / text_content 混用的输出, 任一字段匹配即保留;
                # bbox_label 与画框显示同源, 为两字段均缺失的项补上可匹配的
                # fallback 标签 "bbox" (字段存在时其取值已被前两项覆盖);
                # 非 dict 项原样保留, 由 json_to_pixel_bboxes 的结构校验给出
                # 带期望格式的报错, 而非在此抛裸 AttributeError
                items = [
                    b
                    for b in items
                    if not isinstance(b, dict)
                    or wanted_label
                    in (
                        _normalized_label(b.get("label")),
                        _normalized_label(b.get("text_content")),
                        _normalized_label(bbox_label(b)),
                    )
                ]

            if flat_images:
                curr_img = flat_images[min(i, len(flat_images) - 1)]
                _batch, height, width, _ch = curr_img.shape
                pixel_bboxes = json_to_pixel_bboxes(items, mode, width, height)
                try:
                    # draw_bbox 返回 [1,H,W,C]
                    drawn_images.append(draw_bbox(curr_img[0], pixel_bboxes, [bbox_label(b) for b in items], log_prefix=_PREFIX))
                except Exception as e:
                    logger.warning(_PREFIX + _LOGS["draw_failed_json"].format(i=i + 1, e=e))
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

        logger.info(
            _PREFIX
            + _LOGS["json_to_bboxes_summary"].format(
                json_count=len(json),
                bbox_count=sum(len(group) for group in output_bboxes),
                mode=mode,
                label=label[0].strip(),
            )
        )
        return (output_bboxes, restructured_images)
