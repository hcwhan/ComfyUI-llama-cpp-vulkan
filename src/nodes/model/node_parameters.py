"""采样参数配置节点, 打包成 kwargs dict 供 Instruct 传给 create_chat_completion."""

from ...core.instruct import DEFAULT_SAMPLING_PARAMS as _DEFAULTS
from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG

_TIPS = LANG["nodes"]["model"]["parameters"]["tooltips"]


def _tooltip(name):
    # tooltip 模板的 {default} 按 widget 默认值填充, 与实际生效值同源
    return _TIPS[name].format(default=_DEFAULTS[name])


class llama_cpp_parameters:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        # widget 默认值与 Instruct 未连接 parameters 端口时的生效值同源
        # (core/instruct.py 的 DEFAULT_SAMPLING_PARAMS), 保证两种接法行为一致
        return {
            "required": {
                "max_gen_tokens": (
                    "INT",
                    {
                        "default": _DEFAULTS["max_gen_tokens"],
                        "min": 0,
                        "max": 65536,
                        "step": 1,
                        "tooltip": _tooltip("max_gen_tokens"),
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": _DEFAULTS["top_k"],
                        "min": 0,
                        "max": 1000,
                        "step": 1,
                        "tooltip": _tooltip("top_k"),
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["top_p"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": _tooltip("top_p"),
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["min_p"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": _tooltip("min_p"),
                    },
                ),
                "typical_p": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["typical_p"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": _tooltip("typical_p"),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["temperature"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": _tooltip("temperature"),
                    },
                ),
                "repeat_penalty": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["repeat_penalty"],
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": _tooltip("repeat_penalty"),
                    },
                ),
                # OpenAI/llama.cpp 语义均允许负值(奖励重复), 统一 -2.0 ~ 2.0
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["frequency_penalty"],
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": _tooltip("frequency_penalty"),
                    },
                ),
                "present_penalty": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["present_penalty"],
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.01,
                        "tooltip": _tooltip("present_penalty"),
                    },
                ),
                "mirostat_mode": (
                    "INT",
                    {
                        "default": _DEFAULTS["mirostat_mode"],
                        "min": 0,
                        "max": 2,
                        "step": 1,
                        "tooltip": _tooltip("mirostat_mode"),
                    },
                ),
                "mirostat_eta": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["mirostat_eta"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": _tooltip("mirostat_eta"),
                    },
                ),
                "mirostat_tau": (
                    "FLOAT",
                    {
                        "default": _DEFAULTS["mirostat_tau"],
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": _tooltip("mirostat_tau"),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)

    def process(self, **kwargs):
        return (kwargs,)
