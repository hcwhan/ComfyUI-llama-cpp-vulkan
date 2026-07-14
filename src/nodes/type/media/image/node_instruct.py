"""llama.cpp image Instruct 节点, 图片推理.

- 逐张模式: 逐张推理, 多图结果用 "====== Image N ======" 分隔行拼接输出
  (下游用 Split Instruct Output 拆回列表, JSON to BBoxes 会自动拆分)
- 批量模式: 全部图片并入同一条 user 消息, 一次推理; 多图时缩放到 max_size
"""

import comfy.model_management as mm

from .....core.cqdm import cqdm
from .....core.instruct import llama_cpp_media_instruct_base
from .....core.storage import LLAMA_CPP_STORAGE
from .....shared.logger import logger
from ..encoding import image_content_item, scale_image, tensor_to_uint8

_IMAGE_MODE_EACH = "逐张模式"
_IMAGE_MODE_BATCH = "批量模式"


class llama_cpp_image_instruct(llama_cpp_media_instruct_base):
    MEDIA_WORD = "图像"
    MODALITY = "image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": (cls.MODEL_TYPE,),
                "images": ("IMAGE",),
                **cls.seed_input(),
                **cls.prompt_inputs(),
                "mode": (
                    [_IMAGE_MODE_EACH, _IMAGE_MODE_BATCH],
                    {
                        "default": _IMAGE_MODE_EACH,
                        "tooltip": "逐张模式: 逐张推理, 每张图各得一条结果.\n批量模式: 全部图片并入单次请求 (多图时缩放到 max_size, 单图保持原分辨率).",
                    },
                ),
                "max_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 16384,
                        "step": 64,
                        "tooltip": "批量模式下输入图片的最大边长.\n仅在发送多张图片时生效, 单张图片保持原分辨率.",
                    },
                ),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def _infer_each(self, messages, user_content, images, seed, params, extract_text, watcher):
        image_content = {"type": "image_url", "image_url": {"url": ""}}
        user_content.append(image_content)
        messages.append({"role": "user", "content": user_content})
        logger.info(f"[llama-cpp-vulkan] Start processing {len(images)} images")

        tmp_list = []
        for i, image in enumerate(cqdm(images)):
            if watcher.interrupted or mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            image_content["image_url"] = image_content_item(tensor_to_uint8(image))["image_url"]
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
            # 分隔行格式与 shared/text_utils 的 split_image_results 约定一致
            if len(images) > 1:
                tmp_list.append(f"====== Image {i + 1} ======")
            tmp_list.append(extract_text(output))

        return "\n\n".join(tmp_list)

    def _infer_batch(self, messages, user_content, images, max_size, seed, params, extract_text):
        for image in images:
            if len(images) > 1:
                user_content.append(image_content_item(scale_image(image, max_size)))
            else:
                user_content.append(image_content_item(tensor_to_uint8(image)))
        return self._single_completion(messages, user_content, seed, params, extract_text)

    def process(
        self,
        vlm_model,
        images,
        seed,
        preset_prompt,
        custom_prompt,
        system_prompt,
        mode,
        max_size,
        strip_thinking,
        force_offload,
        parameters=None,
        queue_handler=None,
    ):
        # queue_handler 仅靠连线本身控制执行顺序, 值无用途; del 显式声明有意不使用
        del queue_handler

        def runner(messages, user_content, seed, params, extract_text, watcher):
            self.require_mmproj("Image")
            if mode == _IMAGE_MODE_BATCH:
                return self._infer_batch(messages, user_content, images, max_size, seed, params, extract_text)
            return self._infer_each(messages, user_content, images, seed, params, extract_text, watcher)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
