"""llama.cpp image Instruct 节点, 图片推理.

- 逐张模式 (Per-Image): 逐张推理, 多图结果用 "======== Image N ========" 前缀行拼接输出
  (下游用 Split Instruct Output 拆回列表, JSON to BBoxes 会自动拆分);
  increment_seed 开启时第 i 张图 (0 起) 以 seed+i 为种子, 使相同图片也能得到不同结果
- 批量模式 (Batch): 全部图片并入同一条 user 消息, 一次推理; 多图时缩放到 max_size
"""

import comfy.model_management as mm

from .....core.cqdm import cqdm
from .....core.instruct import llama_cpp_media_instruct_base
from .....i18n.common_static import IMAGE_MODE_BATCH, IMAGE_MODE_EACH, IMAGE_RESULT_SEPARATOR_TEMPLATE, LOG_PREFIX
from .....i18n.lang import LANG
from .....shared.encoding import image_content_item, scale_image, tensor_to_uint8
from .....shared.logger import logger

_TIPS = LANG["nodes"]["instruct"]["image"]["tooltips"]


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
                # each_mode_value/batch_mode_value 是自定义 key, 经 /object_info
                # 原样透传给前端 JS (web/image_instruct.js): increment_seed 仅在 Per-Image 档显示, max_size 仅在 Batch 档显示, 与常量单一真源
                "mode": (
                    [IMAGE_MODE_EACH, IMAGE_MODE_BATCH],
                    {
                        "default": IMAGE_MODE_EACH,
                        "tooltip": _TIPS["mode"],
                        "each_mode_value": IMAGE_MODE_EACH,
                        "batch_mode_value": IMAGE_MODE_BATCH,
                    },
                ),
                "increment_seed": ("BOOLEAN", {"default": False, "tooltip": _TIPS["increment_seed"]}),
                "max_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 16384,
                        "step": 64,
                        "tooltip": _TIPS["max_size"],
                    },
                ),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def _infer_each(
        self,
        messages,
        user_content,
        images,
        seed,
        increment_seed,
        params,
        extract_text,
        watcher,
    ):
        # 同一 messages 列表跨多次请求复用, 每轮仅原位改写 image_content 的
        # url: 依赖 handler 渲染消息时不改写 messages 入参 (当前 wheel 的
        # MTMD handler 满足; 行为性假设无法静态锁定, 已记入 AGENTS.md
        # 对接面清单, 升级 wheel 时人工复核)
        image_content = {"type": "image_url", "image_url": {"url": ""}}
        user_content.append(image_content)
        messages.append({"role": "user", "content": user_content})
        logger.info(LOG_PREFIX + LANG["logs"]["image_instruct"]["start_processing"].format(count=len(images)))

        tmp_list = []
        for i, image in enumerate(cqdm(images)):
            if watcher.interrupted or mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            image_content["image_url"] = image_content_item(tensor_to_uint8(image))["image_url"]
            # 取模 0xFFFFFFFF 使派生值回绕到 [0, 0xFFFFFFFE], 避开 llama.cpp
            # 的随机种子哨兵值 0xFFFFFFFF (语义见 seed_input 的注释)
            request_seed = (seed + i) % 0xFFFFFFFF if increment_seed else seed
            output = self._completion_with_stats(messages, request_seed, params)
            # 前缀行模板与 shared/text_utils 的拆分正则同源 (common_static)
            if len(images) > 1:
                tmp_list.append(IMAGE_RESULT_SEPARATOR_TEMPLATE.format(n=i + 1))
            tmp_list.append(extract_text(output))

        return "\n\n".join(tmp_list)

    def _infer_batch(
        self,
        messages,
        user_content,
        images,
        seed,
        max_size,
        params,
        extract_text,
    ):
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
        increment_seed,
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
            if mode == IMAGE_MODE_EACH:
                return self._infer_each(messages, user_content, images, seed, increment_seed, params, extract_text, watcher)
            if mode == IMAGE_MODE_BATCH:
                return self._infer_batch(messages, user_content, images, seed, max_size, params, extract_text)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
