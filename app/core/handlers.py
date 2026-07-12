"""Chat handler 注册表.

requirements.txt 固定的 JamePeng 分支 wheel 把全部 VLM handler 集中在
llama_multimodal 模块. 缺失的类只会让对应选项从下拉框消失
(启动日志给出 warning), 不会静默吞掉错误.
"""

import functools

import llama_cpp.llama_multimodal as _handler_module

from ..shared.logger import logger

# 显示名 -> (类名, 构造 kwargs)。kwargs 为构造 handler 时固定注入的参数
# (thinking 开关、Generic 的 chat_format 等), None 表示不注入任何参数;
# 每个 key 必须被该类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError)。
# 带 "-Thinking" 后缀的显示名与基名共享同一个类, 仅 thinking 值不同,
# 后缀与开关值的一致性由 tests/test_handlers.py 契约测试锁定。
# dict 声明顺序即 UI 下拉框顺序。新增 handler 只需在此加一行。
_HANDLER_SPECS = {
    "LLaVA-1.5": ("Llava15ChatHandler", None),
    "LLaVA-1.6": ("Llava16ChatHandler", None),
    "Obsidian": ("ObsidianChatHandler", None),
    "Moondream2": ("MoondreamChatHandler", None),
    "nanoLLaVA": ("NanoLlavaChatHandler", None),
    "llama3-Vision-Alpha": ("Llama3VisionAlphaChatHandler", None),
    "MiniCPM-v2.6": ("MiniCPMv26ChatHandler", None),
    "DeepSeek-OCR": ("MTMDChatHandler", None),
    # Gemma4 的 enable_thinking 按 wheel 说明仅 31B/26BA4B 支持,
    # E2B/E4B 输出异常时建议选 -Thinking 变体(等于旧行为)
    "Gemma3": ("Gemma3ChatHandler", None),
    "Gemma4": ("Gemma4ChatHandler", {"enable_thinking": False}),
    "Gemma4-Thinking": ("Gemma4ChatHandler", {"enable_thinking": True}),
    "Qwen2.5-VL": ("Qwen25VLChatHandler", None),
    "MinerU2.5-Pro": ("Qwen25VLChatHandler", None),
    "Qwen3-VL": ("Qwen3VLChatHandler", {"force_reasoning": False}),
    "Qwen3-VL-Thinking": ("Qwen3VLChatHandler", {"force_reasoning": True}),
    "Qwen3.5": ("Qwen35ChatHandler", {"enable_thinking": False}),
    "Qwen3.5-Thinking": ("Qwen35ChatHandler", {"enable_thinking": True}),
    "Qwen3.6": ("Qwen35ChatHandler", {"enable_thinking": False}),
    "Qwen3.6-Thinking": ("Qwen35ChatHandler", {"enable_thinking": True}),
    "GLM-4.6V": ("GLM46VChatHandler", {"enable_thinking": False}),
    "GLM-4.6V-Thinking": ("GLM46VChatHandler", {"enable_thinking": True}),
    # GLM41VChatHandler 不接受 enable_thinking(模板固定输出 thinking 块),
    # 名字的 -Thinking 仅描述模型本身
    "GLM-4.1V-Thinking": ("GLM41VChatHandler", None),
    "LFM2-VL": ("LFM2VLChatHandler", None),
    "LFM2.5-VL": ("LFM25VLChatHandler", None),
    "Granite-Docling": ("GraniteDoclingChatHandler", None),
    "MiniCPM-v4.5": ("MiniCPMv45ChatHandler", {"enable_thinking": False}),
    "MiniCPM-v4.5-Thinking": ("MiniCPMv45ChatHandler", {"enable_thinking": True}),
    "MiniCPM-v4.6": ("MiniCPMV46ChatHandler", {"enable_thinking": False}),
    "MiniCPM-v4.6-Thinking": ("MiniCPMV46ChatHandler", {"enable_thinking": True}),
    "PaddleOCR-VL-1.5": ("PaddleOCRChatHandler", None),
    "Qwen3-ASR": ("Qwen3ASRChatHandler", None),
    "Step3-VL": ("Step3VLChatHandler", {"enable_thinking": False}),
    "Step3-VL-Thinking": ("Step3VLChatHandler", {"enable_thinking": True}),
    # 兜底 handler:渲染 GGUF 内置 chat template 并归一化媒体占位符,
    # 适配上表没有专用 handler 的 VLM;需要特殊 stop token/生成参数的
    # 模型仍应优先用专用 handler。chat_format=None 表示沿用模型内置模板
    "Generic-MTMD": ("GenericMTMDChatHandler", {"chat_format": None}),
}


def _resolve_handlers():
    available = {}
    missing = []
    for label, (cls_name, kwargs) in _HANDLER_SPECS.items():
        handler_cls = getattr(_handler_module, cls_name, None)
        if handler_cls is None:
            missing.append(f"{label} ({cls_name})")
            continue
        if kwargs:
            handler_cls = functools.partial(handler_cls, **kwargs)
        available[label] = handler_cls
    if missing:
        logger.warning(f"[llama-cpp-vulkan] chat handler(s) unavailable in this llama-cpp-python build: {', '.join(missing)}")
    return available


HANDLERS = _resolve_handlers()
