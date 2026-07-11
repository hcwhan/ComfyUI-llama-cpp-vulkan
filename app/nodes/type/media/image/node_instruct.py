"""llama.cpp image Instruct 节点, 图片推理.

- 逐张模式: 逐张推理, 每张各得一条结果(output_list 供下游列表节点消费)
- 批量模式: 全部图片并入同一条 user 消息, 一次推理; 多图时缩放到 max_size
"""

import comfy.model_management as mm

from .....core.cqdm import cqdm
from .....core.instruct import llama_cpp_media_instruct_base
from .....core.storage import LLAMA_CPP_STORAGE
from ..encoding import image_content_item, scale_image, tensor_to_uint8


class llama_cpp_image_instruct(llama_cpp_media_instruct_base):
    MEDIA_WORD = "图像"
    PRESETS = [
        "常规 - 描述",
        "提示词风格 - 标签",
        "提示词风格 - 简洁",
        "提示词风格 - 详细",
        "提示词风格 - 极致详细",
        "提示词风格 - 电影感",
        "创意 - 详细分析",
        "创意 - 短篇故事",
        "视觉 - *BBox 检测",
        "空白 - 自定义",
    ]
    DEFAULT_PRESET = "常规 - 描述"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": (cls.MODEL_TYPE,),
                "images": ("IMAGE",),
                **cls.prompt_inputs(),
                "batch_images": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "关: 逐张推理, 每张图各得一条结果.\n开: 全部图片并入单次请求, 图片会缩放到 max_size."
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": "批量模式下输入图片的最大边长.\n仅在发送多张图片时生效, 单张图片保持原分辨率."
                }),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def _infer_one_by_one(self, messages, user_content, images, seed, params, extract_text, watcher):
        image_content = {"type": "image_url", "image_url": {"url": ""}}
        user_content.append(image_content)
        messages.append({"role": "user", "content": user_content})
        print(f"[llama-cpp-vulkan] Start processing {len(images)} images")

        out_list = []
        tmp_list = []
        for i, image in enumerate(cqdm(images)):
            if watcher.interrupted or mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            image_content["image_url"] = image_content_item(tensor_to_uint8(image))["image_url"]
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
            text = extract_text(output)
            out_list.append(text)
            if len(images) > 1:
                tmp_list.append(f"====== Image {i+1} ======")
            tmp_list.append(text)

        return "\n\n".join(tmp_list), out_list

    def _infer_batch(self, messages, user_content, images, max_size, seed, params, extract_text):
        for image in images:
            if len(images) > 1:
                user_content.append(image_content_item(scale_image(image, max_size)))
            else:
                user_content.append(image_content_item(tensor_to_uint8(image)))
        return self._single_completion(messages, user_content, seed, params, extract_text)

    def process(self, vlm_model, images, preset_prompt, custom_prompt, system_prompt, batch_images, max_size, seed, force_offload, strip_thinking, parameters=None, queue_handler=None):
        def runner(messages, user_content, seed, params, extract_text, watcher):
            self.require_mmproj("Image")
            if batch_images:
                return self._infer_batch(messages, user_content, images, max_size, seed, params, extract_text)
            return self._infer_one_by_one(messages, user_content, images, seed, params, extract_text, watcher)

        return self._run(vlm_model, preset_prompt, custom_prompt, system_prompt, seed, force_offload, strip_thinking, parameters, runner)
