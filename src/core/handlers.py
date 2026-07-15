"""Chat handler 注册表.

requirements.txt 固定的 JamePeng 分支 wheel 把全部 VLM handler 集中在
llama_multimodal 模块. 缺失的类只会让对应选项从下拉框消失
(启动日志给出 warning), 不会静默吞掉错误.
"""

import functools

import llama_cpp.llama_multimodal as _handler_module

from ..i18n.common_static import LOG_PREFIX
from ..i18n.lang import LANG
from ..shared.logger import logger

_LOGS = LANG["logs"]["handlers"]

# thinking 三态: 可切换档直接记 handler 构造参数名(enable_thinking /
# Qwen3-VL 的 force_reasoning), 另两档用哨兵常量; 钳制规则见 clamp_thinking.
# 哨兵消费点统一按相等比较(==/in/dict-get), 勿用 is(字符串同一性依赖
# CPython 驻留实现细节)
THINK_UNSUPPORTED = "unsupported"  # 不支持思考: thinking 恒按关处理
THINK_FORCED = "forced"  # 强制思考(模板固定输出思考块): thinking 恒按开处理

# 显示名 -> (类名, 构造 kwargs, thinking 三态). kwargs 为构造 handler 时固定
# 注入的参数(Generic 的 chat_format 等), None 表示不注入任何参数;
# 每个 key 必须被该类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError).
# thinking 开关值由 VLM Loader 的 thinking 字段提供, 经 handler_constructor
# 按三态绑定; 三态元数据与类签名的一致性由 tests/test_handlers.py 契约锁定.
# dict 声明顺序即 UI 下拉框顺序(首项为默认选项), 排序约定: 同家族聚组,
# 组内按模型发布时间从新到旧; 常用家族在前, OCR/文档专用类居中,
# 早期模型在后, -Generic- 兜底收尾.
_HANDLER_SPECS = {
    # ---- Qwen ----
    # Qwen3.6 与 Qwen3.5 共用 handler (wheel 明确该类同时服务两者)
    "Qwen3.6": ("Qwen35ChatHandler", None, "enable_thinking"),
    "Qwen3.5": ("Qwen35ChatHandler", None, "enable_thinking"),
    # 显示名带 "(ASR) " 前缀, 在下拉框中标注音频(语音识别)模型
    "(ASR) Qwen3-ASR": ("Qwen3ASRChatHandler", None, THINK_UNSUPPORTED),
    "Qwen3-VL": ("Qwen3VLChatHandler", None, "force_reasoning"),
    "Qwen2.5-VL": ("Qwen25VLChatHandler", None, THINK_UNSUPPORTED),
    # ---- GLM ----
    "GLM-4.6V": ("GLM46VChatHandler", None, "enable_thinking"),
    # GLM41VChatHandler 不接受思考开关参数 (模板固定输出 thinking 块),
    # "-Thinking" 后缀描述模型本身
    "GLM-4.1V-Thinking": ("GLM41VChatHandler", None, THINK_FORCED),
    # ---- Gemma ----
    # Gemma4 的 enable_thinking 按 wheel 说明仅 31B/26BA4B 支持;
    # E2B 实测: 传 False 安全, 但 E 系列关不掉思考(仍以纯文本思考并输出
    # <channel|> 分隔符), 思考内容由 strip_thinking 按 <channel|> 剥离
    "Gemma4": ("Gemma4ChatHandler", None, "enable_thinking"),
    "Gemma3": ("Gemma3ChatHandler", None, THINK_UNSUPPORTED),
    # ---- MiniCPM ----
    "MiniCPM-v4.6": ("MiniCPMV46ChatHandler", None, "enable_thinking"),
    "MiniCPM-v4.5": ("MiniCPMv45ChatHandler", None, "enable_thinking"),
    "MiniCPM-v2.6": ("MiniCPMv26ChatHandler", None, THINK_UNSUPPORTED),
    # ---- Step ----
    "Step3-VL": ("Step3VLChatHandler", None, "enable_thinking"),
    # ---- LFM ----
    "LFM2.5-VL": ("LFM25VLChatHandler", None, THINK_UNSUPPORTED),
    "LFM2-VL": ("LFM2VLChatHandler", None, THINK_UNSUPPORTED),
    # ---- OCR / 文档专用 ----
    # 显示名带 "(OCR) " 前缀, 在下拉框中与通用视觉模型区分
    # MinerU2.5-Pro 基于 Qwen2.5-VL, 复用其 handler
    "(OCR) MinerU2.5-Pro": ("Qwen25VLChatHandler", None, THINK_UNSUPPORTED),
    "(OCR) PaddleOCR-VL-1.5": ("PaddleOCRChatHandler", None, THINK_UNSUPPORTED),
    "(OCR) DeepSeek-OCR": ("MTMDChatHandler", None, THINK_UNSUPPORTED),
    "(OCR) Granite-Docling": ("GraniteDoclingChatHandler", None, THINK_UNSUPPORTED),
    # ---- 早期模型 ----
    "llama3-Vision-Alpha": ("Llama3VisionAlphaChatHandler", None, THINK_UNSUPPORTED),
    "nanoLLaVA": ("NanoLlavaChatHandler", None, THINK_UNSUPPORTED),
    "Moondream2": ("MoondreamChatHandler", None, THINK_UNSUPPORTED),
    "LLaVA-1.6": ("Llava16ChatHandler", None, THINK_UNSUPPORTED),
    "LLaVA-1.5": ("Llava15ChatHandler", None, THINK_UNSUPPORTED),
    "Obsidian": ("ObsidianChatHandler", None, THINK_UNSUPPORTED),
    # ---- 兜底 ----
    # 渲染 GGUF 内置 chat template 并归一化媒体占位符,
    # 适配上表没有专用 handler 的 VLM; 需要特殊 stop token/生成参数的
    # 模型仍应优先用专用 handler. chat_format=None 表示沿用模型内置模板
    "-Generic-": ("GenericMTMDChatHandler", {"chat_format": None}, THINK_UNSUPPORTED),
}


def _resolve_handlers():
    available = {}
    missing = []
    for label, (cls_name, kwargs, _think) in _HANDLER_SPECS.items():
        handler_cls = getattr(_handler_module, cls_name, None)
        if handler_cls is None:
            missing.append(f"{label} ({cls_name})")
            continue
        if kwargs:
            handler_cls = functools.partial(handler_cls, **kwargs)
        available[label] = handler_cls
    if missing:
        logger.warning(LOG_PREFIX + _LOGS["handlers_unavailable"].format(missing=", ".join(missing)))
    return available


HANDLERS = _resolve_handlers()


def clamp_thinking(label, thinking):
    """按三态钳制 thinking 请求值, 返回实际生效值.

    不支持档收到 True 按关处理, 强制档收到 False 按开处理, 各打 warning.
    这是前端 JS 三态置灰的后端兜底, 覆盖 API 提交/旧工作流等绕过 UI 的路径.
    """
    think = _HANDLER_SPECS[label][2]
    if think == THINK_UNSUPPORTED and thinking:
        logger.warning(LOG_PREFIX + _LOGS["thinking_unsupported"].format(label=label))
        return False
    if think == THINK_FORCED and not thinking:
        logger.warning(LOG_PREFIX + _LOGS["thinking_forced"].format(label=label))
        return True
    return thinking


def is_registered(label):
    """label 是否在注册表中声明; 与 HANDLERS 的差集即本构建缺类的 handler.

    供 resolve_config 区分 "未知名字" 与 "注册过但 wheel 缺类" 两种报错.
    """
    return label in _HANDLER_SPECS


def handler_constructor(label, thinking):
    """返回按三态绑定 thinking 后的 handler 构造器 (label 未知时抛 KeyError).

    可切换档把开关值绑到声明的参数名上; 不支持/强制档的类没有开关参数,
    原样返回构造器 (强制档的思考由模板固化, 无需传参).
    """
    ctor = HANDLERS[label]
    think = _HANDLER_SPECS[label][2]
    if think in (THINK_UNSUPPORTED, THINK_FORCED):
        return ctor
    return functools.partial(ctor, **{think: thinking})


def thinking_modes():
    """label -> "toggle"/"forced"/"none" 能力名单 (只含可用 handler).

    经 VLM Loader 的 chat_handler widget options 透传给前端 JS 做三态置灰,
    与注册表单一真源.
    """
    sentinel = {THINK_UNSUPPORTED: "none", THINK_FORCED: "forced"}
    return {label: sentinel.get(_HANDLER_SPECS[label][2], "toggle") for label in HANDLERS}


# 音频专用 handler: 无视觉编码路径, loader 的 image_min/max_tokens 对其无效
_AUDIO_ONLY_LABELS = frozenset({"(ASR) Qwen3-ASR"})


def image_token_handlers():
    """支持 image_min/max_tokens 的 handler 名单 (视觉类, 音频专用档除外).

    经 VLM Loader 的 chat_handler widget options 透传给前端 JS,
    控制两个 token 字段的显隐; -Generic- 可能服务任意模态, 保守按支持处理.
    """
    return [label for label in HANDLERS if label not in _AUDIO_ONLY_LABELS]
