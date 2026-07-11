"""Instruct 节点共享基类: 消息组装, 推理执行, 中断监视, thinking 剥离, hybrid 重置.

- llama_cpp_instruct_base        文本推理骨架(MODEL_TYPE = LLAMACPPLLM)
- llama_cpp_media_instruct_base  多模态推理骨架(MODEL_TYPE = LLAMACPPVLM, 附 mmproj 校验)

子类(各 node_instruct.py)负责: 声明 INPUT_TYPES(用本类提供的字段组装块 +
模态专属字段), 把媒体内容注入 user_content, 选择执行路径.
"""

import re
import threading

import comfy.model_management as mm

from ..shared.types import any_type
from .prompts import preset_prompts
from .storage import LLAMA_CPP_STORAGE

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking_blocks(text):
    """移除 <think>...</think> 推理块。

    Thinking 模型的 generation prompt 通常已注入开头的 <think>,
    此时输出只含闭合标签,需要取最后一个 </think> 之后的内容。
    """
    if "</think>" not in text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.lstrip()


def is_hybrid_arch(llm):
    """判断模型是否为 hybrid/recurrent 架构(如 Qwen3.5 的线性注意力、Mamba 类)。

    纯 SWA 模型(如 Gemma3)不算:其前缀缓存由 llama-cpp-python 内置的
    checkpoint 机制处理,无需请求后整体重置。
    """
    return llm._model.is_hybrid() or llm._model.is_recurrent()


class InterruptWatcher:
    """推理期间轮询 ComfyUI 的中断标志,命中时触发 llama 的 abort_event。

    create_completion 在每次请求开始时会 clear abort_event,
    因此命中后持续重复 set 而不是设置一次就退出,避免竞态丢失中断。
    """

    def __init__(self, llm, poll_interval=0.2):
        self.llm = llm
        self.poll_interval = poll_interval
        self.interrupted = False
        self._stop = threading.Event()
        self._thread = None

    def _watch(self):
        while not self._stop.wait(self.poll_interval):
            if mm.processing_interrupted():
                self.interrupted = True
                try:
                    self.llm.abort()
                except Exception:
                    pass

    def __enter__(self):
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        return False


class llama_cpp_instruct_base:
    CATEGORY = "llama-cpp-vulkan"
    FUNCTION = "process"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "output_list")
    OUTPUT_IS_LIST = (False, True)

    # 子类覆盖:模型端口类型(llm_model/vlm_model) / 预设模板 @ 占位符替换词 / 可选预设与默认项
    MODEL_TYPE = "LLAMACPPLLM"
    MEDIA_WORD = "图像"
    PRESETS = list(preset_prompts)
    DEFAULT_PRESET = "空白 - 自定义"

    # ---- INPUT_TYPES 字段组装块(子类按需拼接,顺序由子类的声明决定) ----

    @classmethod
    def prompt_inputs(cls):
        return {
            "preset_prompt": (cls.PRESETS, {"default": cls.DEFAULT_PRESET}),
            "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": '用户提示词\n\n预设名带 "*" 时, 此内容用于填充预设中的占位符(如 BBox 检测的目标类别)\n否则, 此内容会整体覆盖预设提示词.'}),
            "system_prompt": ("STRING", {"multiline": True, "default": ""}),
        }

    @classmethod
    def runtime_inputs(cls):
        return {
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "step": 1, "tooltip": "llama.cpp 使用 32 位种子, 更大的值会被截断."}),
            "force_offload": ("BOOLEAN", {
                "default": False,
                "tooltip": "推理结束后立即卸载模型, 释放显存."
            }),
            "strip_thinking": ("BOOLEAN", {
                "default": True,
                "tooltip": "移除输出中的 <think>...</think> 推理块.\n(适用于 Thinking 模型)"
            }),
        }

    @classmethod
    def optional_inputs(cls):
        return {
            "parameters": ("LLAMACPPARAMS",),
            "queue_handler": (any_type, {"tooltip": "用于控制多个 Instruct 节点的执行顺序."}),
        }

    # ---- 执行核心 ----

    def _prepare_messages(self, llama_model, system_prompt):
        """确保目标模型已加载,并构建本次请求的初始消息列表(无跨执行状态)。

        多组 loader+instruct 交错执行时,全局单例可能已被切换成其他模型,
        因此按 current_config 比对后按需(重新)加载。
        """
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model)

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        return messages

    def _build_prompt_text(self, preset_prompt, custom_prompt):
        if custom_prompt.strip() and "*" not in preset_prompt:
            return {"type": "text", "text": custom_prompt}
        if "*" in preset_prompt and not custom_prompt.strip():
            raise ValueError(f'Preset "{preset_prompt}" requires custom_prompt to fill its placeholder (e.g. object categories for BBox detection).')
        # 先替换 @ 再注入用户文本,避免 custom_prompt 中的 @ 被误替换
        p = preset_prompts[preset_prompt].replace("@", self.MEDIA_WORD).replace("#", custom_prompt.strip())
        return {"type": "text", "text": p}

    def _make_extract(self, strip_thinking):
        def extract_text(output):
            text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            return strip_thinking_blocks(text) if strip_thinking else text
        return extract_text

    def _single_completion(self, messages, user_content, seed, params, extract_text):
        """把 user_content 作为单条 user 消息发起一次补全。"""
        messages.append({"role": "user", "content": user_content})
        output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
        out1 = extract_text(output)
        return out1, [out1]

    def _run(self, llama_model, preset_prompt, custom_prompt, system_prompt, seed, force_offload, strip_thinking, parameters, runner):
        """通用执行骨架:组消息 -> 中断监视下执行 runner -> 收尾清理。

        runner(messages, user_content, seed, params, extract_text, watcher)
        由子类提供,返回 (output, output_list)。
        """
        messages = self._prepare_messages(llama_model, system_prompt)
        user_content = [self._build_prompt_text(preset_prompt, custom_prompt)]
        # 先复制再修改,避免污染 ComfyUI 缓存的共享参数 dict
        params = (parameters or {}).copy()
        extract_text = self._make_extract(strip_thinking)

        # 监视线程让长时间生成也能响应 ComfyUI 的取消操作
        with InterruptWatcher(LLAMA_CPP_STORAGE.llm) as watcher:
            out1, out2 = runner(messages, user_content, seed, params, extract_text, watcher)

        if watcher.interrupted:
            # abort_event 使生成提前返回了截断结果,丢弃并走标准中断流程
            raise mm.InterruptProcessingException()

        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        elif is_hybrid_arch(LLAMA_CPP_STORAGE.llm):
            # 真 hybrid/recurrent 架构(Qwen3.5、LFM2 系等)的线性注意力状态无法
            # 跨请求做前缀复用,不重置会导致后续请求输出错乱;按架构判断而非
            # handler 名单,避免每加一个 hybrid 模型都要维护名单
            llm = LLAMA_CPP_STORAGE.llm
            llm.n_tokens = 0
            llm._ctx.memory_clear(True)
            if llm._hybrid_cache_mgr is not None:
                llm._hybrid_cache_mgr.clear()

        return (out1, out2)


class llama_cpp_media_instruct_base(llama_cpp_instruct_base):
    MODEL_TYPE = "LLAMACPPVLM"

    @staticmethod
    def require_mmproj(kind):
        if not getattr(LLAMA_CPP_STORAGE.chat_handler, "mmproj_path", None):
            raise ValueError(f"{kind} input detected, but the loaded model is not configured with a mmproj module.")
