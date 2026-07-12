"""Chat handler 注册表.

requirements.txt 固定的 JamePeng 分支 wheel 把全部 VLM handler 集中在
llama_multimodal 模块. 缺失的类只会让对应选项从下拉框消失
(启动日志给出 warning), 不会静默吞掉错误.
"""

import functools

import llama_cpp.llama_multimodal as _handler_module

from ..shared.logger import logger

# 显示名 -> (类名, thinking 开关参数名)。带 "-Thinking" 后缀的显示名共享同一个
# 类,加载时按后缀切换开关值。新增 handler 只需在此加一行。
# 注意 thinking 参数名必须被该类 __init__ 接受(基类对未知 kwargs 抛 TypeError)。
_HANDLER_SPECS = {
    "LLaVA-1.5": ("Llava15ChatHandler", None),
    "LLaVA-1.6": ("Llava16ChatHandler", None),
    "Moondream2": ("MoondreamChatHandler", None),
    "nanoLLaVA": ("NanoLlavaChatHandler", None),
    "llama3-Vision-Alpha": ("Llama3VisionAlphaChatHandler", None),
    "MiniCPM-v2.6": ("MiniCPMv26ChatHandler", None),
    "DeepSeek-OCR": ("MTMDChatHandler", None),
    "Gemma3": ("Gemma3ChatHandler", None),
    "Gemma4": ("Gemma4ChatHandler", None),
    "Qwen2.5-VL": ("Qwen25VLChatHandler", None),
    "MinerU2.5-Pro": ("Qwen25VLChatHandler", None),
    "Qwen3-VL": ("Qwen3VLChatHandler", "force_reasoning"),
    "Qwen3-VL-Thinking": ("Qwen3VLChatHandler", "force_reasoning"),
    "Qwen3.5": ("Qwen35ChatHandler", "enable_thinking"),
    "Qwen3.5-Thinking": ("Qwen35ChatHandler", "enable_thinking"),
    "Qwen3.6": ("Qwen35ChatHandler", "enable_thinking"),
    "Qwen3.6-Thinking": ("Qwen35ChatHandler", "enable_thinking"),
    "GLM-4.6V": ("GLM46VChatHandler", "enable_thinking"),
    "GLM-4.6V-Thinking": ("GLM46VChatHandler", "enable_thinking"),
    # GLM41VChatHandler 不接受 enable_thinking(模板固定输出 thinking 块)
    "GLM-4.1V-Thinking": ("GLM41VChatHandler", None),
    "LFM2-VL": ("LFM2VLChatHandler", None),
    "LFM2.5-VL": ("LFM25VLChatHandler", None),
    "Granite-Docling": ("GraniteDoclingChatHandler", None),
    "MiniCPM-v4.5": ("MiniCPMv45ChatHandler", "enable_thinking"),
    "MiniCPM-v4.5-Thinking": ("MiniCPMv45ChatHandler", "enable_thinking"),
    "MiniCPM-v4.6": ("MiniCPMV46ChatHandler", "enable_thinking"),
    "MiniCPM-v4.6-Thinking": ("MiniCPMV46ChatHandler", "enable_thinking"),
    "PaddleOCR-VL-1.5": ("PaddleOCRChatHandler", None),
    "Qwen3-ASR": ("Qwen3ASRChatHandler", None),
    "Step3-VL": ("Step3VLChatHandler", None),
    # 兜底 handler:渲染 GGUF 内置 chat template 并归一化媒体占位符,
    # 适配上表没有专用 handler 的 VLM;需要特殊 stop token/生成参数的
    # 模型仍应优先用专用 handler
    "Generic-MTMD": ("GenericMTMDChatHandler", None),
}

# 构造时需要固定注入的参数:Generic 的 chat_format 是必填位置参数,
# None 表示沿用模型内置 chat template
_FIXED_KWARGS = {
    "Generic-MTMD": {"chat_format": None},
}


def _resolve_handlers():
    available = {}
    missing = []
    for label, (cls_name, think_param) in _HANDLER_SPECS.items():
        handler_cls = getattr(_handler_module, cls_name, None)
        if handler_cls is None:
            missing.append(f"{label} ({cls_name})")
            continue
        fixed = _FIXED_KWARGS.get(label)
        if fixed:
            handler_cls = functools.partial(handler_cls, **fixed)
        available[label] = (handler_cls, think_param)
    if missing:
        logger.warning(f"[llama-cpp-vulkan] chat handler(s) unavailable in this llama-cpp-python build: {', '.join(missing)}")
    return available


HANDLERS = _resolve_handlers()
