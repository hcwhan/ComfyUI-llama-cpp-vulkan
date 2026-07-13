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
from .prompts import instruct_presets, preset_content
from .storage import LLAMA_CPP_STORAGE

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Gemma4 思考块的闭合 token (格式 <|channel>thought ... <channel|>)。
# E2B/E4B 在 enable_thinking=False 时仍会以纯文本思考并自行输出 <channel|>
# 分隔符(无开标签, 实测确认), 因此不能按 "开标签...闭标签" 成对匹配,
# 统一取最后一个 <channel|> 之后的内容
_GEMMA4_CHANNEL_CLOSE = "<channel|>"

_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"


def _unwrap_answer(text):
    """剥离 GLM-4.1V 形态的 <answer>...</answer> 包裹。

    GLM-4.1V-Thinking 的输出为 <think>...</think>\\n<answer>正文</answer>
    (官方推理代码按 <answer>(.*?)</answer> 提取正文); 本插件的 handler 以
    </answer> 为 stop token, 闭合标签通常不进入文本, 因此开标签会残留。
    仅在文本以 <answer> 开头时剥离, 避免误伤正文中的同名字样。
    """
    stripped = text.lstrip()
    if not stripped.startswith(_ANSWER_OPEN):
        return text
    stripped = stripped[len(_ANSWER_OPEN):]
    if stripped.rstrip().endswith(_ANSWER_CLOSE):
        stripped = stripped.rstrip()[:-len(_ANSWER_CLOSE)]
    return stripped.strip()


def strip_thinking_blocks(text):
    """移除思考块: <think>...</think>、Gemma4 的 channel 格式、GLM-4.1V 的 <answer> 包裹。

    Thinking 模型的 generation prompt 通常已注入开头的 <think>,
    此时输出只含闭合标签,需要取最后一个 </think> 之后的内容。
    Gemma4 同理只认闭合 token <channel|>; 未闭合(生成截断)时保持原样。
    """
    if "</think>" in text:
        cleaned = _THINK_BLOCK_RE.sub("", text)
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[-1]
        text = cleaned.lstrip()
    if _GEMMA4_CHANNEL_CLOSE in text:
        text = text.rsplit(_GEMMA4_CHANNEL_CLOSE, 1)[-1].lstrip()
    return _unwrap_answer(text)


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
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)

    # 子类覆盖:模型端口类型(llm_model/vlm_model) / 预设模板 @@@ 占位符替换词 /
    # 模态标识(按预设的 use 字段过滤下拉框名单, 列表第一项即默认预设)
    MODEL_TYPE = "LLAMACPPLLM"
    MEDIA_WORD = "图像"
    MODALITY = "text"

    # ---- INPUT_TYPES 字段组装块(子类按需拼接,顺序由子类的声明决定) ----

    @classmethod
    def prompt_inputs(cls):
        presets = instruct_presets(cls.MODALITY)
        return {
            "preset_prompt": (presets, {"default": presets[0]}),
            "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": '用户提示词\n\n预设含占位符时(如 BBox 检测的目标类别、待改写的提示词), 此内容用于填充占位符\n否则, 此内容会整体覆盖预设提示词.'}),
            "system_prompt": ("STRING", {"multiline": True, "default": ""}),
        }

    @classmethod
    def runtime_inputs(cls):
        return {
            # 上限取 0xFFFFFFFE: 0xFFFFFFFF 是 llama.cpp 的 LLAMA_DEFAULT_SEED
            # 哨兵值(随机种子), 落在该值会静默失去可复现性
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xfffffffe, "step": 1, "tooltip": "32 位种子; 上限 0xFFFFFFFE, 避开 llama.cpp 的随机种子哨兵值 0xFFFFFFFF."}),
            "force_offload": ("BOOLEAN", {
                "default": False,
                "tooltip": "推理结束后立即卸载模型, 释放显存."
            }),
            "strip_thinking": ("BOOLEAN", {
                "default": True,
                "tooltip": "移除输出中的思考/推理块\n(适用于 Thinking 模型)"
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

    def _build_user_prompt(self, preset_prompt, custom_prompt):
        """构建 user 消息的文本内容项(模板取自 user_prompt_presets)。

        覆盖/填充按模板内容判定:
        - 模板含 "###":custom_prompt 是填充物(必填),替换 "###" 占位符
        - 模板不含 "###":非空 custom_prompt 整体覆盖预设,为空则用模板原文
        预设名的 "(需custom_prompt)" 标注仅为 UI 提示,不参与判定。
        """
        template = preset_content(preset_prompt)
        if "###" not in template:
            if custom_prompt.strip():
                return {"type": "text", "text": custom_prompt}
        elif not custom_prompt.strip():
            raise ValueError(f'Preset "{preset_prompt}" requires custom_prompt to fill its placeholder (e.g. object categories for BBox detection, or the prompt to rewrite).')
        # 先替换 @@@ 再注入用户文本,避免 custom_prompt 中的 @@@ 被误替换
        p = template.replace("@@@", self.MEDIA_WORD).replace("###", custom_prompt.strip())
        return {"type": "text", "text": p}

    def _make_extract(self, strip_thinking):
        def extract_text(output):
            # ": " 前缀剥离针对 Vicuna/LLaVA 风格模板(生成提示以 "ASSISTANT:" 收尾,
            # 如 Llava15ChatHandler 与部分 GGUF 内嵌模板): 部分模型会把冒号连同空格
            # 再输出一遍。正文本身以 ": " 开头的场景极罕见, 误剥按可接受代价处理
            text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            return strip_thinking_blocks(text) if strip_thinking else text
        return extract_text

    def _single_completion(self, messages, user_content, seed, params, extract_text):
        """把 user_content 作为单条 user 消息发起一次补全,返回生成文本。

        content 只含单个 text 项时扁平化为纯字符串:无 chat handler 的纯文本路径
        由 GGUF 内嵌 chat template 渲染消息,旧式模板(ChatML/Llama-3/Mistral 等)
        假定 content 是字符串,收到 content-part 列表会报错或渲染出 Python repr。
        媒体路径的 content 必然追加了媒体项(长度 > 1),不受影响。
        """
        if len(user_content) == 1 and user_content[0].get("type") == "text":
            user_content = user_content[0]["text"]
        messages.append({"role": "user", "content": user_content})
        output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
        return extract_text(output)

    def _run(self, llama_model, preset_prompt, custom_prompt, system_prompt, seed, force_offload, strip_thinking, parameters, runner):
        """通用执行骨架:组消息 -> 中断监视下执行 runner -> 收尾清理。

        runner(messages, user_content, seed, params, extract_text, watcher)
        由子类提供,返回输出文本(image 逐张模式为按分隔行拼接的整段文本,
        下游可用 Split Instruct Output 节点或 JSON to BBoxes 的内建拆分还原)。
        """
        # 先做零成本的 prompt 校验(占位符预设缺 custom_prompt 时直接 ValueError),
        # 再触发可能长达数 GB 的模型加载, 避免漏填时白白完成一次全量加载才报错
        user_content = [self._build_user_prompt(preset_prompt, custom_prompt)]
        messages = self._prepare_messages(llama_model, system_prompt)
        # 防御性复制:parameters 是 ComfyUI 缓存的共享 dict,防止 runner 修改时污染
        params = (parameters or {}).copy()
        extract_text = self._make_extract(strip_thinking)

        # 监视线程让长时间生成也能响应 ComfyUI 的取消操作;
        # 收尾放 finally:中断/异常路径同样需要 force_offload 释放显存与 hybrid 重置
        try:
            with InterruptWatcher(LLAMA_CPP_STORAGE.llm) as watcher:
                out = runner(messages, user_content, seed, params, extract_text, watcher)
            if watcher.interrupted:
                # abort_event 使生成提前返回了截断结果,丢弃并走标准中断流程
                raise mm.InterruptProcessingException()
        finally:
            if force_offload:
                LLAMA_CPP_STORAGE.clean()
            elif LLAMA_CPP_STORAGE.llm is not None and is_hybrid_arch(LLAMA_CPP_STORAGE.llm):
                # 真 hybrid/recurrent 架构(Qwen3.5、LFM2 系等)的线性注意力状态无法
                # 跨请求做前缀复用,不重置会导致后续请求输出错乱;按架构判断而非
                # handler 名单,避免每加一个 hybrid 模型都要维护名单
                llm = LLAMA_CPP_STORAGE.llm
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()

        return (out,)


class llama_cpp_media_instruct_base(llama_cpp_instruct_base):
    MODEL_TYPE = "LLAMACPPVLM"

    @staticmethod
    def require_mmproj(kind):
        if not getattr(LLAMA_CPP_STORAGE.chat_handler, "mmproj_path", None):
            raise ValueError(f"{kind} input detected, but the loaded model is not configured with a mmproj module.")
