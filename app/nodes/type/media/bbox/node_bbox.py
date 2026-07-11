"""BBox 工具链节点: JSON 解析转框, SEGS/MASK 转换, 索引选取."""

import torch
import numpy as np

from .....shared.text_utils import parse_json
from .bbox_utils import (
    QWEN_BBOX_MODES,
    bbox_label,
    draw_bbox,
    feathered_rect_mask,
    json_to_pixel_bboxes,
    valid_int_bbox,
)


class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper

    def __repr__(self):
        return (f"SEG(cropped_image={self.cropped_image}, cropped_mask=shape{self.cropped_mask.shape}, confidence={self.confidence}, bbox={self.bbox}, label='{self.label}'), control_net_wrapper={self.control_net_wrapper}")


class json_to_bboxes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "只保留指定 label 的 BBox."
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
        label = label[0]

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
            print(f"[llama-cpp-vulkan] Warning: {len(json)} JSON result(s) but {len(flat_images)} image frame(s); pairing by index, extra entries reuse the last frame")

        output_bboxes = []
        drawn_images = []

        for i, json_str in enumerate(json):
            items = parse_json(json_str)
            if label != "":
                # 兼容 label / text_content 混用的输出,任一字段匹配即保留
                items = [b for b in items if label in (b.get("label"), b.get("text_content"))]

            if flat_images:
                curr_img = flat_images[min(i, len(flat_images) - 1)]
                _batch, height, width, _ch = curr_img.shape
                pixel_bboxes = json_to_pixel_bboxes(items, mode, width, height)
                try:
                    # draw_bbox 返回 [1,H,W,C]
                    drawn_images.append(draw_bbox(curr_img[0], pixel_bboxes, [bbox_label(b) for b in items]))
                except Exception as e:
                    print(f"Error drawing bboxes for JSON #{i}: {e}")
                    drawn_images.append(curr_img)
            else:
                pixel_bboxes = json_to_pixel_bboxes(items, mode)

            output_bboxes.append(pixel_bboxes)

        # 画框结果与 JSON 条目一一对应,按输入图像的批次结构重新分组
        restructured_images = []
        cursor = 0
        for count in batch_sizes:
            chunk = drawn_images[cursor:cursor + count]
            if chunk:
                restructured_images.append(torch.cat(chunk, dim=0))
            cursor += count

        return (output_bboxes, restructured_images)


class bboxes_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)

        seg_list = []
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
                print(f"Warning: Skipping bbox outside image bounds: {bbox}")
                continue

            # 扩张区域同样限制在图像内，保证 crop_region 不含负坐标（Impact Pack 约定）
            x1_exp = max(0, x1 - dilation)
            y1_exp = max(0, y1 - dilation)
            x2_exp = min(width, x2 + dilation)
            y2_exp = min(height, y2 + dilation)

            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp

            # 原始 bbox 在扩张窗口坐标系中的位置
            inner_rect = (x1 - x1_exp, y1 - y1_exp, x2 - x1_exp, y2 - y1_exp)
            cropped_mask_np = feathered_rect_mask(crop_h, crop_w, inner_rect, feather)
            # Impact Pack 的 SEG 约定 cropped_image 为 [1, H, W, C]
            cropped_image_tensor = image_for_cropping[y1_exp:y2_exp, x1_exp:x2_exp, :].unsqueeze(0)

            seg = SEG(
                cropped_image=cropped_image_tensor,
                cropped_mask=cropped_mask_np,
                # Impact Pack 约定 confidence 为标量
                confidence=0.9,
                crop_region=crop_region,
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                label="bbox"
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
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200, "step": 1}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
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
                continue

            # 局部窗口(含羽化边界),裁剪到图像范围
            wx1, wy1 = max(0, x1_exp - margin), max(0, y1_exp - margin)
            wx2, wy2 = min(width, x2_exp + margin), min(height, y2_exp + margin)
            if wx2 <= wx1 or wy2 <= wy1:
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
