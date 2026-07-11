import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from .shared import parse_json, draw_bbox, qwen3bbox


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


class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {
                    "default":"",
                    "multiline": False,
                    "tooltip": "Select only the BBoxes with specific labels."
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
        mode = mode[0]
        label = label[0]

        flat_images_list = []
        original_structure = []

        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])

        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []

        for i, j in enumerate(json):
            bboxes = parse_json(j)

            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]

                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]

                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)

            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]

            output_bboxes.append(bbox)

        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count

        return (output_bboxes, restructured_images_list)


class bbox_to_segs:
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
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)

        seg_list = []
        image_for_cropping = image[0]

        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue

            x1, y1, x2, y2 = map(int, bbox)
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

            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_x1 = x1 - x1_exp
            local_y1 = y1 - y1_exp
            local_x2 = x2 - x1_exp
            local_y2 = y2 - y1_exp
            local_mask_np[local_y1:local_y2, local_x1:local_x2] = 1.0

            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)

            cropped_mask_np = local_mask_np
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


class bbox_to_mask:
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

        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                print(f"Warning: Skipping invalid bbox item: {bbox}")
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1_exp = x1 - dilation
            y1_exp = y1 - dilation
            x2_exp = x2 + dilation
            y2_exp = y2 + dilation
            crop_w = x2_exp - x1_exp
            crop_h = y2_exp - y1_exp

            if crop_h <= 0 or crop_w <= 0:
                continue

            current_full_mask_np = np.zeros(mask_shape, dtype=np.float32)
            x1_c, y1_c = max(0, x1_exp), max(0, y1_exp)
            x2_c, y2_c = min(width, x2_exp), min(height, y2_exp)

            if x2_c > x1_c and y2_c > y1_c:
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = 1.0

            if feather > 0:
                current_full_mask_np = gaussian_filter(current_full_mask_np, sigma=feather)

            current_full_mask_tensor = torch.from_numpy(current_full_mask_np).to(image.device)
            combined_full_mask = torch.maximum(combined_full_mask, current_full_mask_tensor)

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
                    "tooltip": "BBox index in the image. Set to 999 to get all bboxes."
                }),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    # 上游 json_to_bbox 的 BBOX 输出是 OUTPUT_IS_LIST（每元素一组）。
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
