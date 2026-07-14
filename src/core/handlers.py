"""Chat handler 注册表.

requirements.txt 固定的 JamePeng 分支 wheel 把全部 VLM handler 集中在
llama_multimodal 模块. 缺失的类只会让对应选项从下拉框消失
(启动日志给出 warning), 不会静默吞掉错误.
"""

import functools

import llama_cpp.llama_multimodal as _handler_module

from ..shared.logger import logger

# 显示名 -> (类名, 构造 kwargs). kwargs 为构造 handler 时固定注入的参数
# (thinking 开关, Generic 的 chat_format 等), None 表示不注入任何参数;
# 每个 key 必须被该类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError).
# 带 "-Thinking" 后缀的显示名与基名共享同一个类, 仅 thinking 值不同,
# 后缀与开关值的一致性由 tests/test_handlers.py 契约测试锁定.
# dict 声明顺序即 UI 下拉框顺序(首项为默认选项), 排序约定: 同家族聚组,
# 组内按模型发布时间从新到旧, -Thinking 变体紧随普通版之下; 常用家族在前,
# OCR/文档专用类居中, 早期模型在后, Generic-MTMD 兜底收尾.
_HANDLER_SPECS = {
    # ---- Qwen ----
    # Qwen3.6 与 Qwen3.5 共用 handler (wheel 明确该类同时服务两者)
    "Qwen3.6": ("Qwen35ChatHandler", {"enable_thinking": False}),
    "Qwen3.6-Thinking": ("Qwen35ChatHandler", {"enable_thinking": True}),
    "Qwen3.5": ("Qwen35ChatHandler", {"enable_thinking": False}),
    "Qwen3.5-Thinking": ("Qwen35ChatHandler", {"enable_thinking": True}),
    # 显示名带 "(ASR) " 前缀, 在下拉框中标注音频(语音识别)模型
    "(ASR) Qwen3-ASR": ("Qwen3ASRChatHandler", None),
    "Qwen3-VL": ("Qwen3VLChatHandler", {"force_reasoning": False}),
    "Qwen3-VL-Thinking": ("Qwen3VLChatHandler", {"force_reasoning": True}),
    "Qwen2.5-VL": ("Qwen25VLChatHandler", None),
    # ---- GLM ----
    "GLM-4.6V": ("GLM46VChatHandler", {"enable_thinking": False}),
    "GLM-4.6V-Thinking": ("GLM46VChatHandler", {"enable_thinking": True}),
    # GLM41VChatHandler 不接受 enable_thinking 参数 (模板固定输出 thinking 块)
    "GLM-4.1V-Thinking": ("GLM41VChatHandler", None),
    # ---- Gemma ----
    # Gemma4 的 enable_thinking 按 wheel 说明仅 31B/26BA4B 支持;
    # E2B 实测: 传 False 安全, 但 E 系列关不掉思考(仍以纯文本思考并输出
    # <channel|> 分隔符), 思考内容由 strip_thinking 按 <channel|> 剥离
    "Gemma4": ("Gemma4ChatHandler", {"enable_thinking": False}),
    "Gemma4-Thinking": ("Gemma4ChatHandler", {"enable_thinking": True}),
    "Gemma3": ("Gemma3ChatHandler", None),
    # ---- MiniCPM ----
    "MiniCPM-v4.6": ("MiniCPMV46ChatHandler", {"enable_thinking": False}),
    "MiniCPM-v4.6-Thinking": ("MiniCPMV46ChatHandler", {"enable_thinking": True}),
    "MiniCPM-v4.5": ("MiniCPMv45ChatHandler", {"enable_thinking": False}),
    "MiniCPM-v4.5-Thinking": ("MiniCPMv45ChatHandler", {"enable_thinking": True}),
    "MiniCPM-v2.6": ("MiniCPMv26ChatHandler", None),
    # ---- Step ----
    "Step3-VL": ("Step3VLChatHandler", {"enable_thinking": False}),
    "Step3-VL-Thinking": ("Step3VLChatHandler", {"enable_thinking": True}),
    # ---- LFM ----
    "LFM2.5-VL": ("LFM25VLChatHandler", None),
    "LFM2-VL": ("LFM2VLChatHandler", None),
    # ---- OCR / 文档专用 ----
    # 显示名带 "(OCR) " 前缀, 在下拉框中与通用视觉模型区分
    # MinerU2.5-Pro 基于 Qwen2.5-VL, 复用其 handler
    "(OCR) MinerU2.5-Pro": ("Qwen25VLChatHandler", None),
    "(OCR) PaddleOCR-VL-1.5": ("PaddleOCRChatHandler", None),
    "(OCR) DeepSeek-OCR": ("MTMDChatHandler", None),
    "(OCR) Granite-Docling": ("GraniteDoclingChatHandler", None),
    # ---- 早期模型 ----
    "llama3-Vision-Alpha": ("Llama3VisionAlphaChatHandler", None),
    "nanoLLaVA": ("NanoLlavaChatHandler", None),
    "Moondream2": ("MoondreamChatHandler", None),
    "LLaVA-1.6": ("Llava16ChatHandler", None),
    "LLaVA-1.5": ("Llava15ChatHandler", None),
    "Obsidian": ("ObsidianChatHandler", None),
    # ---- 兜底 ----
    # 渲染 GGUF 内置 chat template 并归一化媒体占位符,
    # 适配上表没有专用 handler 的 VLM; 需要特殊 stop token/生成参数的
    # 模型仍应优先用专用 handler. chat_format=None 表示沿用模型内置模板
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
