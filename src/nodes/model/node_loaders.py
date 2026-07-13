"""模型加载节点: llm(纯文本)与 vlm(多模态)两个 Loader.

两者只做快速失败校验并输出配置 dict, 实际加载由 Instruct 节点按需触发
(懒加载: 多组 loader+instruct 交错时避免全局单例被 loader 反复挤占).

输出类型完全隔离:
- llm Loader -> LLAMACPPLLM, 只能连 text Instruct
- vlm Loader -> LLAMACPPVLM, 只能连 image/video/audio Instruct
"""

import os

from ...core.devices import AUTO_LABEL, gpu_device_choices
from ...core.handlers import HANDLERS
from ...core.model_paths import get_llm_filename_list
from ...core.storage import resolve_config

_GPU_DEVICE_FIELD = (gpu_device_choices, {
    "default": AUTO_LABEL,
    "tooltip": "选择 LLM 推理使用的 GPU 设备.\nAuto = llama.cpp 默认行为: 独显优先, 多独显按层切分.\n显式选择某设备时, 整个模型加载到该单卡.\n(仅当系统没有独显时, 核显才可选)"
})

_N_CTX_FIELD = ("INT", {
    "default": 8192,
    "min": 1024, "max": 327680, "step": 128,
    "tooltip": "上下文长度上限."
})

_VRAM_LIMIT_FIELD = ("INT", {
    "default": -1,
    "min": -1, "max": 1024, "step": 1,
    "tooltip": "显存占用上限, 单位 GB.\n-1 = 自动 (llama.cpp 按空闲显存适配层数), 0 = 纯 CPU 推理.\n预算不足模型 1 层时全部留在 CPU (严格守上限);\n层体积为估算值, 实际占用可能略有偏差."
})


def _is_mmproj(path):
    # 只看文件名,避免目录名含 mmproj 时整个目录的主模型被误过滤
    return "mmproj" in os.path.basename(path).lower()


def _model_list():
    return ["None"] + [f for f in get_llm_filename_list() if not _is_mmproj(f)]


def _mmproj_list():
    # 无 mmproj 文件时保留 "None" 占位,避免空下拉框;运行期校验会给出明确报错
    return [f for f in get_llm_filename_list() if _is_mmproj(f)] or ["None"]


class llama_cpp_llm_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "gpu_device": _GPU_DEVICE_FIELD,
            "model": (_model_list(),),
            "n_ctx": _N_CTX_FIELD,
            "vram_limit": _VRAM_LIMIT_FIELD,
        }}

    RETURN_TYPES = ("LLAMACPPLLM",)
    RETURN_NAMES = ("llm_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vulkan"

    def loadmodel(self, gpu_device, model, n_ctx, vram_limit):
        if model == "None":
            raise ValueError("Please select a gguf model.")
        config = {
            "model": model,
            "mmproj": "None",
            "chat_handler": "None",
            "gpu_device": gpu_device,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": 0,
            "image_max_tokens": 0,
        }
        resolve_config(config)
        return (config,)


class llama_cpp_vlm_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "gpu_device": _GPU_DEVICE_FIELD,
            "model": (_model_list(),),
            "mmproj": (_mmproj_list(),),
            # "None" 占位在首位作为默认值, 强制用户显式选择匹配的 handler
            # (loadmodel 做非空校验), 避免默认首个 handler 被误用于不匹配的模型
            "chat_handler": (["None"] + list(HANDLERS),),
            "n_ctx": _N_CTX_FIELD,
            "vram_limit": _VRAM_LIMIT_FIELD,
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
        }}

    RETURN_TYPES = ("LLAMACPPVLM",)
    RETURN_NAMES = ("vlm_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vulkan"

    def loadmodel(self, gpu_device, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if model == "None":
            raise ValueError("Please select a gguf model.")
        if mmproj == "None":
            raise ValueError("vlm Model Loader requires a mmproj file. Put the matching mmproj gguf in the llm/LLM folder, or use llm Model Loader for text-only models.")
        if chat_handler == "None":
            raise ValueError("vlm Model Loader requires a chat handler matching the model.")
        # 与 handler 侧同一条件,只是提前到 loader 报错(<=0 视为未设置)
        if 0 < image_max_tokens < image_min_tokens:
            raise ValueError(f"image_max_tokens ({image_max_tokens}) cannot be less than image_min_tokens ({image_min_tokens}).")
        config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "gpu_device": gpu_device,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
        }
        resolve_config(config)
        return (config,)
