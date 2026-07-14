"""模型加载节点: llm(纯文本)与 vlm(多模态)两个 Loader.

两者只做快速失败校验并输出配置 dict, 实际加载由 Instruct 节点按需触发
(懒加载: 多组 loader+instruct 交错时避免全局单例被 loader 反复挤占).

输出类型完全隔离:
- llm Loader -> LLAMACPPLLM, 只能连 text Instruct
- vlm Loader -> LLAMACPPVLM, 只能连 image/video/audio Instruct
"""

import os

from ...core.devices import AUTO_LABEL, gpu_device_choices
from ...core.handlers import HANDLERS, clamp_thinking, image_token_handlers, thinking_modes
from ...core.model_paths import get_llm_filename_list
from ...core.storage import resolve_config

_GPU_DEVICE_FIELD = (
    gpu_device_choices,
    {
        "default": AUTO_LABEL,
        "tooltip": "选择 LLM 推理使用的 GPU 设备.\nAuto = llama.cpp 默认行为: 独显优先, 多独显按层切分.\n显式选择某设备时, 整个模型加载到该单卡.\n(仅当系统没有独显时, 核显才可选)",
    },
)

_CTX_SIZE_FIELD = (
    "INT",
    {
        "default": 8192,
        "min": 1024,
        "max": 327680,
        "step": 128,
        "tooltip": "上下文长度上限, 即 llama.cpp 的 n_ctx.\n请求的 prompt + 生成 token 总量受此约束.",
    },
)

_VRAM_LIMIT_FIELD = (
    "INT",
    {
        "default": -1,
        "min": -1,
        "max": 1024,
        "step": 1,
        "tooltip": "显存占用上限, 单位 GB.\n-1 = 自动 (llama.cpp 按空闲显存适配层数), 0 = 纯 CPU 推理.\n预算不足模型 1 层时全部留在 CPU (严格守上限);\n层体积为估算值, 实际占用可能略有偏差.",
    },
)


def _is_mmproj(path):
    # 只看文件名, 避免目录名含 mmproj 时整个目录的主模型被误过滤
    return "mmproj" in os.path.basename(path).lower()


def _model_list():
    return ["None"] + [f for f in get_llm_filename_list() if not _is_mmproj(f)]


def _mmproj_list():
    # "None" 占位在首位作为默认值, 强制用户显式选择与主模型配对的 mmproj
    # (loadmodel 做非空校验), 与 model / chat_handler 的显式选择原则统一:
    # 静默选中首个文件时, 错配要到 mtmd 首次推理才报错, 距配置点太远
    return ["None"] + [f for f in get_llm_filename_list() if _is_mmproj(f)]


class llama_cpp_llm_model_loader:
    CATEGORY = "llama-cpp-vulkan"
    FUNCTION = "loadmodel"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_device": _GPU_DEVICE_FIELD,
                "model": (_model_list(),),
                "ctx_size": _CTX_SIZE_FIELD,
                "vram_limit": _VRAM_LIMIT_FIELD,
            }
        }

    RETURN_TYPES = ("LLAMACPPLLM",)
    RETURN_NAMES = ("llm_model",)

    def loadmodel(self, gpu_device, model, ctx_size, vram_limit):
        if model == "None":
            raise ValueError("Please select a gguf model.")
        config = {
            "gpu_device": gpu_device,
            "model": model,
            "mmproj": "None",
            "chat_handler": "None",
            "thinking": False,
            "n_ctx": ctx_size,
            "vram_limit": vram_limit,
            "image_min_tokens": 0,
            "image_max_tokens": 0,
        }
        resolve_config(config)
        return (config,)


class llama_cpp_vlm_model_loader:
    CATEGORY = "llama-cpp-vulkan"
    FUNCTION = "loadmodel"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_device": _GPU_DEVICE_FIELD,
                "model": (_model_list(),),
                "mmproj": (_mmproj_list(),),
                # "None" 占位在首位作为默认值, 强制用户显式选择匹配的 handler
                # (loadmodel 做非空校验), 避免默认首个 handler 被误用于不匹配的模型.
                # thinking_modes / image_token_handlers 是自定义 key, 经 /object_info
                # 原样透传给前端 JS: 前者做 thinking 开关的三态置灰, 后者控制
                # image_min/max_tokens 的显隐(均与注册表单一真源)
                "chat_handler": (
                    ["None"] + list(HANDLERS),
                    {"thinking_modes": thinking_modes(), "image_token_handlers": image_token_handlers()},
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "开启模型的思考(reasoning)模式.\n构造期模板级开关: 切换后下次执行会整体重新加载模型.\n仅对支持切换的 handler 生效: 不支持思考的强制为关,\nGLM-4.1V 等纯思考模型强制为开.\nGemma4 E2B/E4B 关闭后仍会以纯文本形式思考,\n残留思考内容由 Instruct 的 strip_thinking 剥离.",
                    },
                ),
                "ctx_size": _CTX_SIZE_FIELD,
                "vram_limit": _VRAM_LIMIT_FIELD,
                "image_min_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "mmproj 视觉编码的最小 token 数.\n0 = 使用模型默认 (<=0 视为未设置).\n仅对图像/视频输入生效, 音频不受影响.\n修改后 Qwen2.5-VL 的 bbox 坐标换算会有偏差.",
                    },
                ),
                "image_max_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "mmproj 视觉编码的最大 token 数, 可限制高分辨率图片的显存与耗时.\n0 = 使用模型默认 (<=0 视为未设置).\n仅对图像/视频输入生效, 音频不受影响.\n修改后 Qwen2.5-VL 的 bbox 坐标换算会有偏差.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LLAMACPPVLM",)
    RETURN_NAMES = ("vlm_model",)

    def loadmodel(self, gpu_device, model, mmproj, chat_handler, thinking, ctx_size, vram_limit, image_min_tokens, image_max_tokens):
        if model == "None":
            raise ValueError("Please select a gguf model.")
        if mmproj == "None":
            raise ValueError(
                "vlm Model Loader requires a mmproj file. Put the matching mmproj gguf in the llm/LLM folder, or use llm Model Loader for text-only models."
            )
        if chat_handler == "None":
            raise ValueError("vlm Model Loader requires a chat handler matching the model.")
        # 与 handler 侧同一条件, 只是提前到 loader 报错(<=0 视为未设置)
        if 0 < image_max_tokens < image_min_tokens:
            raise ValueError(f"image_max_tokens ({image_max_tokens}) cannot be less than image_min_tokens ({image_min_tokens}).")
        config = {
            "gpu_device": gpu_device,
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "thinking": thinking,
            "n_ctx": ctx_size,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
        }
        resolve_config(config)
        # 校验通过后钳制并落盘实际生效值(warning 离配置点近), 也避免不可切换档
        # 的开关值变化引起无意义的 current_config 失配重载
        config["thinking"] = clamp_thinking(chat_handler, thinking)
        return (config,)
