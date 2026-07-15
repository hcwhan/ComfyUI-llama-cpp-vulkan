"""模型加载节点: LLM(纯文本)与 VLM(多模态)两个 Loader.

两者只做快速失败校验并输出配置 dict, 实际加载由 Instruct 节点按需触发
(懒加载: 多组 loader+instruct 交错时避免全局单例被 loader 反复挤占).

输出类型完全隔离:
- LLM Loader -> LLAMACPPLLM, 只能连 text Instruct
- VLM Loader -> LLAMACPPVLM, 只能连 image/video/audio Instruct
"""

import os

from ...core.devices import gpu_device_choices
from ...core.handlers import HANDLERS, clamp_thinking, image_token_handlers, thinking_modes
from ...core.model_paths import get_llm_filename_list
from ...core.storage import resolve_config
from ...i18n.common_static import AUTO_LABEL, NONE_OPTION
from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG

_COMMON_TIPS = LANG["nodes"]["model"]["common"]["tooltips"]
_COMMON_ERRORS = LANG["nodes"]["model"]["common"]["errors"]
_LLM_TIPS = LANG["nodes"]["model"]["llm_loader"]["tooltips"]
_VLM_TIPS = LANG["nodes"]["model"]["vlm_loader"]["tooltips"]
_VLM_ERRORS = LANG["nodes"]["model"]["vlm_loader"]["errors"]

_GPU_DEVICE_FIELD = (
    gpu_device_choices,
    {
        "default": AUTO_LABEL,
        "tooltip": _COMMON_TIPS["gpu_device"],
    },
)

# UI 名 ctx_size; config dict 内部键维持 wheel 术语 n_ctx (Llama(n_ctx=...)).
# vlm 默认取 llm 的 2 倍: 图像/视频帧经 mmproj 编码的媒体 token 直接占用
# 上下文, 同样的文本余量需要更大的 n_ctx
_CTX_SIZE_DEFAULT = 8192


def _ctx_size_field(default):
    return (
        "INT",
        {
            "default": default,
            "min": 1024,
            "max": 327680,
            "step": 128,
            "tooltip": _COMMON_TIPS["ctx_size"],
        },
    )


# 数值语义两个 Loader 相同, tooltip 分开维护 (VLM 侧多 mmproj 预算扣除语义)
def _vram_limit_field(tooltip):
    return (
        "INT",
        {
            "default": -1,
            "min": -1,
            "max": 1024,
            "step": 1,
            "tooltip": tooltip,
        },
    )


def _is_mmproj(path):
    # 只看文件名, 避免目录名含 mmproj 时整个目录的主模型被误过滤
    return "mmproj" in os.path.basename(path).lower()


def _model_list():
    return [NONE_OPTION] + [f for f in get_llm_filename_list() if not _is_mmproj(f)]


def _mmproj_list():
    # "None" 占位在首位作为默认值, 强制用户显式选择与主模型配对的 mmproj
    # (loadmodel 做非空校验), 与 model / chat_handler 的显式选择原则统一:
    # 静默选中首个文件时, 错配要到 mtmd 首次推理才报错, 距配置点太远
    return [NONE_OPTION] + [f for f in get_llm_filename_list() if _is_mmproj(f)]


class llama_cpp_llm_model_loader:
    CATEGORY = _CATEGORY
    FUNCTION = "loadmodel"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_device": _GPU_DEVICE_FIELD,
                "model": (_model_list(),),
                "ctx_size": _ctx_size_field(_CTX_SIZE_DEFAULT),
                "vram_limit": _vram_limit_field(_LLM_TIPS["vram_limit"]),
            }
        }

    RETURN_TYPES = ("LLAMACPPLLM",)
    RETURN_NAMES = ("llm_model",)

    def loadmodel(self, gpu_device, model, ctx_size, vram_limit):
        if model == NONE_OPTION:
            raise ValueError(_COMMON_ERRORS["model_not_selected"])
        config = {
            "gpu_device": gpu_device,
            "model": model,
            "mmproj": NONE_OPTION,
            "chat_handler": NONE_OPTION,
            "thinking": False,
            "n_ctx": ctx_size,
            "vram_limit": vram_limit,
            "image_min_tokens": 0,
            "image_max_tokens": 0,
        }
        resolve_config(config)
        return (config,)


class llama_cpp_vlm_model_loader:
    CATEGORY = _CATEGORY
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
                    [NONE_OPTION] + list(HANDLERS),
                    {"thinking_modes": thinking_modes(), "image_token_handlers": image_token_handlers()},
                ),
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": _VLM_TIPS["thinking"],
                    },
                ),
                "ctx_size": _ctx_size_field(_CTX_SIZE_DEFAULT * 2),
                "vram_limit": _vram_limit_field(_VLM_TIPS["vram_limit"]),
                "image_min_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": _VLM_TIPS["image_min_tokens"],
                    },
                ),
                "image_max_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": _VLM_TIPS["image_max_tokens"],
                    },
                ),
            }
        }

    RETURN_TYPES = ("LLAMACPPVLM",)
    RETURN_NAMES = ("vlm_model",)

    def loadmodel(self, gpu_device, model, mmproj, chat_handler, thinking, ctx_size, vram_limit, image_min_tokens, image_max_tokens):
        if model == NONE_OPTION:
            raise ValueError(_COMMON_ERRORS["model_not_selected"])
        if mmproj == NONE_OPTION:
            raise ValueError(_VLM_ERRORS["mmproj_not_selected"])
        if chat_handler == NONE_OPTION:
            raise ValueError(_VLM_ERRORS["handler_not_selected"])
        # 无视觉编码路径的 handler (音频专用) 把两个 image token 参数折算为 0
        # (0 = 未设置): 与前端隐藏字段的行为对应, widget 值本身保留不动,
        # 重新显示时不丢失; 也顺带豁免下方的区间校验 (隐藏字段无法在 UI 修正)
        if chat_handler not in image_token_handlers():
            image_min_tokens = 0
            image_max_tokens = 0
        # 与 handler 侧同一条件, 只是提前到 loader 报错(<=0 视为未设置)
        if 0 < image_max_tokens < image_min_tokens:
            raise ValueError(_VLM_ERRORS["image_token_range"].format(image_max_tokens=image_max_tokens, image_min_tokens=image_min_tokens))
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
