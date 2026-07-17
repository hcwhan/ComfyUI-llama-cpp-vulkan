"""Instruct 节点共享基类: 消息组装, 推理执行, 生成统计日志, 中断监视, thinking 剥离, hybrid 重置.

- llama_cpp_instruct_base        文本推理骨架(MODEL_TYPE = LLAMACPPLLM)
- llama_cpp_media_instruct_base  多模态推理骨架(MODEL_TYPE = LLAMACPPVLM, 附 mmproj 校验)

子类(各 node_instruct.py)负责: 声明 INPUT_TYPES(用本类提供的字段组装块 +
模态专属字段), 把媒体内容注入 user_content, 选择执行路径.
"""

import contextlib
import functools
import re
import threading
import time

import comfy.model_management as mm
from llama_cpp import llama_chat_format

from ..i18n.common_static import CATEGORY as _CATEGORY
from ..i18n.common_static import LOG_PREFIX
from ..i18n.lang import LANG
from ..shared.logger import logger, node_log_prefix
from ..shared.types import any_type
from .prompts import instruct_presets, preset_content
from .storage import LLAMA_CPP_STORAGE

_COMMON = LANG["nodes"]["instruct"]["common"]
_TIPS = _COMMON["tooltips"]
_PLACEHOLDERS = _COMMON["placeholders"]
_ERRORS = _COMMON["errors"]
_LOGS = LANG["logs"]["instruct"]

# 采样参数的统一默认值: Parameters 节点的 widget 默认值与 Instruct 未连接
# parameters 端口时的生效值均取自此表, 保证连不连默认参数节点行为一致
# (否则未连接时会落到 wheel 库签名默认值: temperature 0.2 (节点默认 0.8) /
# repeat_penalty 1.0 (节点默认 1.1) 等, 与节点默认差异明显).
DEFAULT_SAMPLING_PARAMS = {
    "max_gen_tokens": 0,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "typical_p": 1.0,
    "temperature": 0.8,
    "repeat_penalty": 1.1,
    "frequency_penalty": 0.0,
    "present_penalty": 0.0,
    "mirostat_mode": 0,
    "mirostat_eta": 0.1,
    "mirostat_tau": 5.0,
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Gemma4 思考块的闭合 token (格式 <|channel>thought ... <channel|>).
# E2B/E4B 在 enable_thinking=False 时仍会以纯文本思考并自行输出 <channel|>
# 分隔符(无开标签, 实测确认), 因此不能按 "开标签...闭标签" 成对匹配,
# 统一取最后一个 <channel|> 之后的内容
_GEMMA4_CHANNEL_CLOSE = "<channel|>"

_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"


def _unwrap_answer(text):
    """剥离 GLM-4.1V 形态的 <answer>...</answer> 包裹.

    GLM-4.1V-Thinking 的输出为 <think>...</think>\\n<answer>正文</answer>
    (官方推理代码按 <answer>(.*?)</answer> 提取正文); 本插件的 handler 以
    </answer> 为 stop token, 闭合标签通常不进入文本, 因此开标签会残留.
    仅在文本以 <answer> 开头时剥离, 避免误伤正文中的同名字样.
    """
    stripped = text.lstrip()
    if not stripped.startswith(_ANSWER_OPEN):
        return text
    stripped = stripped[len(_ANSWER_OPEN) :]
    if stripped.rstrip().endswith(_ANSWER_CLOSE):
        stripped = stripped.rstrip()[: -len(_ANSWER_CLOSE)]
    return stripped.strip()


def strip_thinking_blocks(text):
    """移除思考块: <think>...</think>, Gemma4 的 channel 格式, GLM-4.1V 的 <answer> 包裹.

    Thinking 模型的 generation prompt 通常已注入开头的 <think>,
    此时输出只含闭合标签, 需要取最后一个 </think> 之后的内容.
    Gemma4 同理只认闭合 token <channel|>; 未闭合(生成截断)时保持原样.
    """
    if "</think>" in text:
        cleaned = _THINK_BLOCK_RE.sub("", text)
        if "</think>" in cleaned:
            cleaned = cleaned.rsplit("</think>", 1)[-1]
        text = cleaned.lstrip()
    if _GEMMA4_CHANNEL_CLOSE in text:
        text = text.rsplit(_GEMMA4_CHANNEL_CLOSE, 1)[-1].lstrip()
    return _unwrap_answer(text)


def _log_completion_stats(output, elapsed, log_prefix):
    """记录单次 completion 的生成统计: 生成 token 数, 用时, 速度, 思考/答案占比.

    prompt_tokens/completion_tokens 取 wheel 统计的 usage 字段 (真实计数,
    对接面见 AGENTS.md 依赖版本对接原则); elapsed 是整次请求的墙钟时间,
    含 prompt 预填充, prompt 很长时速度会低于纯解码速度.

    思考/答案拆分: 把剥离思考块后的答案文本重新 tokenize 计数, 思考部分取
    与 completion_tokens 的差值. 独立 tokenize 与生成时在上下文中的切分
    可能有个位数 token 出入, 两个数值均为估算 (日志文案带 "约"); 未检出
    思考形态 (剥离前后文本相同) 时不打印拆分. 拆分不受 strip_thinking
    开关影响: 统计的是本次生成实际花在思考上的量, 与输出后处理无关.

    log_prefix 由调用方按节点名构造 (node_log_prefix), 统计行随所属节点标识.
    """
    usage = output["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    speed = completion_tokens / elapsed if elapsed > 0 else 0.0
    content = output["choices"][0]["message"]["content"]
    answer = strip_thinking_blocks(content)
    if answer == content:
        logger.info(
            log_prefix
            + _LOGS["generation_stats"].format(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, elapsed=elapsed, speed=speed
            )
        )
        return
    answer_tokens = len(LLAMA_CPP_STORAGE.llm.tokenize(answer.encode("utf-8"), add_bos=False, special=False))
    logger.info(
        log_prefix
        + _LOGS["generation_stats_thinking"].format(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=max(completion_tokens - answer_tokens, 0),
            answer_tokens=answer_tokens,
            elapsed=elapsed,
            speed=speed,
        )
    )


@functools.lru_cache(maxsize=4)
def _template_preinjects_think(template):
    """渲染 GGUF 内嵌 chat template, 判断 generation prompt 尾部是否预注入 <think>.

    复用 wheel 的 Jinja2ChatFormatter 保证与运行时同一渲染路径 (同款沙箱环境与
    raise_exception/strftime_now 全局, add_generation_prompt 取构造默认 True);
    bos/eos 传空串, 模板引用处渲染为空, 不影响尾部是否以 <think> 收尾的判定.
    渲染失败 (模板对消息形态有本探测未满足的要求等) 按未预注入处理,
    行为与不探测时一致; 结果按模板字符串缓存, 每个模板仅渲染一次.
    """
    try:
        formatter = llama_chat_format.Jinja2ChatFormatter(template=template, eos_token="", bos_token="")
        prompt = formatter(messages=[{"role": "user", "content": "probe"}]).prompt
    except Exception as e:
        logger.debug(LOG_PREFIX + _LOGS["think_probe_failed"].format(e=e))
        return False
    return prompt.rstrip().endswith("<think>")


def think_open_preinjected(llm):
    """判断已加载模型的文本路径 generation prompt 是否已预注入 <think> 开标签.

    预注入形态 (Qwen3.5 等) 下 wheel 的 reasoning_budget 采样器等不到生成的
    开标签, 需要调用方改传 reasoning_start_in_prompt=True; 本函数为 text
    Instruct 提供该判定. 仅 chat_format 落在 "chat_template.default"
    (GGUF 内嵌模板) 时渲染探测: 插件从不显式指定 chat_format, 文本路径
    其余可达值只有 wheel 猜中的内置格式 (chatml/llama-3/mistral-instruct)
    与 fallback (llama-2), 均不预注入 <think>, 直接判 False.
    """
    if llm.chat_format != "chat_template.default":
        return False
    return _template_preinjects_think(llm.metadata["tokenizer.chat_template"])


def is_hybrid_arch(llm):
    """判断模型是否为 hybrid/recurrent 架构(如 Qwen3.5 的线性注意力, Mamba 类).

    纯 SWA 模型(如 Gemma3)不算: 其前缀缓存由 llama-cpp-python 内置的
    checkpoint 机制处理, 无需请求后整体重置.
    """
    return llm._model.is_hybrid() or llm._model.is_recurrent()


class InterruptWatcher:
    """推理期间轮询 ComfyUI 的中断标志, 命中时触发 llama 的 abort_event.

    create_completion 在每次请求开始时会 clear abort_event,
    因此命中后持续重复 set 而不是设置一次就退出, 避免竞态丢失中断;
    命中日志只在首次置位时打一条. log_prefix 由调用方按节点名传入
    (node_log_prefix), 默认纯插件前缀.
    """

    def __init__(self, llm, poll_interval=0.2, log_prefix=LOG_PREFIX):
        self.llm = llm
        self.poll_interval = poll_interval
        self.log_prefix = log_prefix
        self.interrupted = False
        self._stop = threading.Event()
        self._thread = None

    def _watch(self):
        while not self._stop.wait(self.poll_interval):
            if mm.processing_interrupted():
                if not self.interrupted:
                    self.interrupted = True
                    logger.info(self.log_prefix + _LOGS["interrupted"])
                with contextlib.suppress(Exception):
                    self.llm.abort()

    def __enter__(self):
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        return False


class llama_cpp_instruct_base:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)

    # 子类覆盖: 模型端口类型(llm_model/vlm_model) / 预设模板 @@@ 占位符替换词 /
    # 模态标识(按预设的 use 字段过滤下拉框名单, 列表第一项即默认预设) /
    # 日志前缀节点名(功能性标识, 不随语言切换)
    MODEL_TYPE = "LLAMACPPLLM"
    MEDIA_WORD = "图像"
    MODALITY = "text"
    LOG_NAME = "Text Instruct"
    # 纯文本路径无媒体载荷, 最终 user 文本为空的请求只会让模型自由发挥,
    # 在 _run 中直接拦截; media 基类关闭(空文本 + 媒体内容是有意设计,
    # 适合 chat 模板自带默认指令的模型)
    REQUIRE_USER_TEXT = True

    # ---- INPUT_TYPES 字段组装块(子类按需拼接, 顺序由子类的声明决定) ----

    @classmethod
    def seed_input(cls):
        return {
            # 上限取 0xFFFFFFFE: 0xFFFFFFFF 是 llama.cpp 的 LLAMA_DEFAULT_SEED
            # 哨兵值(随机种子), 落在该值会静默失去可复现性
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFE,
                    "step": 1,
                    "tooltip": _TIPS["seed"],
                },
            ),
        }

    @classmethod
    def prompt_inputs(cls):
        presets = instruct_presets(cls.MODALITY)
        return {
            "preset_prompt": (presets, {"default": presets[0]}),
            "custom_prompt": (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "placeholder": _PLACEHOLDERS["custom_prompt"],
                },
            ),
            "system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": _PLACEHOLDERS["system_prompt"]}),
        }

    @classmethod
    def runtime_inputs(cls):
        # force_offload 是执行结束后的收尾动作, 垫底; 输出后处理 (strip_thinking) 靠前
        return {
            "strip_thinking": ("BOOLEAN", {"default": True, "tooltip": _TIPS["strip_thinking"]}),
            "force_offload": ("BOOLEAN", {"default": True, "tooltip": _TIPS["force_offload"]}),
        }

    @classmethod
    def optional_inputs(cls):
        return {
            "parameters": ("LLAMACPPARAMS", {"tooltip": _TIPS["parameters"]}),
            "queue_handler": (any_type, {"tooltip": _TIPS["queue_handler"]}),
        }

    # ---- 执行核心 ----

    def _prepare_messages(self, llama_model, system_prompt):
        """确保目标模型已加载, 并构建本次请求的初始消息列表(无跨执行状态).

        多组 loader+instruct 交错执行时, 全局单例可能已被切换成其他模型,
        因此按 current_config 比对后按需(重新)加载.
        """
        if LLAMA_CPP_STORAGE.llm is None or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model)
        else:
            logger.info(node_log_prefix(self.LOG_NAME) + _LOGS["model_reused"])

        messages = []
        system_prompt = system_prompt.strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        return messages

    def _build_user_prompt(self, preset_prompt, custom_prompt):
        """构建 user 消息的文本内容项(模板取自 user_prompt_presets).

        覆盖/填充按模板内容判定:
        - 模板含 "###": custom_prompt 是填充物(必填), 替换 "###" 占位符
        - 模板不含 "###": 非空 custom_prompt 整体覆盖预设, 为空则用模板原文
        预设名的 "(需custom_prompt)" 标注仅为 UI 提示, 不参与判定.
        """
        template = preset_content(preset_prompt)
        if "###" not in template:
            if custom_prompt.strip():
                # strip 与填充分支的注入形态一致, 首尾空白不随请求发送
                return {"type": "text", "text": custom_prompt.strip()}
        elif not custom_prompt.strip():
            raise ValueError(_ERRORS["preset_requires_custom_prompt"].format(preset_prompt=preset_prompt))
        # 先替换 @@@ 再注入用户文本, 避免 custom_prompt 中的 @@@ 被误替换
        p = template.replace("@@@", self.MEDIA_WORD).replace("###", custom_prompt.strip())
        return {"type": "text", "text": p}

    def _make_extract(self, strip_thinking):
        def extract_text(output):
            # ": " 前缀剥离针对 Vicuna/LLaVA 风格模板(生成提示以 "ASSISTANT:" 收尾,
            # 如 Llava15ChatHandler 与部分 GGUF 内嵌模板): 部分模型会把冒号连同空格
            # 再输出一遍. 正文本身以 ": " 开头的场景极罕见, 误剥按可接受代价处理
            text = output["choices"][0]["message"]["content"].removeprefix(": ").lstrip()
            return strip_thinking_blocks(text) if strip_thinking else text

        return extract_text

    def _completion_with_stats(self, messages, seed, params):
        """发起一次 completion 并记录生成统计日志, 返回原始 output dict.

        全部 create_chat_completion 调用点(_single_completion 与 image
        逐张模式)统一经此包装, 保证每次请求都有一条统计日志.
        """
        start = time.perf_counter()
        output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
        _log_completion_stats(output, time.perf_counter() - start, node_log_prefix(self.LOG_NAME))
        return output

    def _single_completion(self, messages, user_content, seed, params, extract_text):
        """把 user_content 作为单条 user 消息发起一次 completion, 返回生成文本.

        content 只含单个 text 项时扁平化为纯字符串: 无 chat handler 的纯文本路径
        由 GGUF 内嵌 chat template 渲染消息, 旧式模板(ChatML/Llama-3/Mistral 等)
        假定 content 是字符串, 收到 content-part 列表会报错或渲染出 Python repr.
        媒体路径的 content 必然追加了媒体项(长度 > 1), 不受影响.
        """
        if len(user_content) == 1 and user_content[0].get("type") == "text":
            user_content = user_content[0]["text"]
        messages.append({"role": "user", "content": user_content})
        output = self._completion_with_stats(messages, seed, params)
        return extract_text(output)

    def _run(self, llama_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner):
        """通用执行骨架: 组消息 -> 中断监视下执行 runner -> 收尾清理.

        runner(messages, user_content, seed, params, extract_text, watcher)
        由子类提供, 返回输出文本(image 逐张模式为按前缀行拼接的整段文本,
        下游可用 Split Instruct Output 节点或 JSON to BBoxes 的内建拆分还原).
        """
        # 先做零成本的 prompt 校验(占位符预设缺 custom_prompt 时直接 ValueError),
        # 再触发可能长达数 GB 的模型加载, 避免漏填时白白完成一次全量加载才报错
        user_content = [self._build_user_prompt(preset_prompt, custom_prompt)]
        if self.REQUIRE_USER_TEXT and not user_content[0]["text"].strip():
            raise ValueError(_ERRORS["user_prompt_empty"])
        # 请求摘要在触发模型加载前打印, 与后续的加载/统计日志衔接成完整链路;
        # 字符数按 strip 后计 (与实际注入的内容一致)
        logger.info(
            node_log_prefix(self.LOG_NAME)
            + _LOGS["request"].format(
                seed=seed,
                preset=preset_prompt,
                custom_chars=len(custom_prompt.strip()),
                system_chars=len(system_prompt.strip()),
                strip_thinking=strip_thinking,
                force_offload=force_offload,
            )
        )
        messages = self._prepare_messages(llama_model, system_prompt)
        # 合并生成新 dict 兼作防御性复制(parameters 是 ComfyUI 缓存的共享 dict,
        # 防止 runner 修改时污染); 未连接 parameters 端口时整体落到统一默认值
        params = {**DEFAULT_SAMPLING_PARAMS, **(parameters or {})}
        params["max_tokens"] = params.pop("max_gen_tokens")
        extract_text = self._make_extract(strip_thinking)

        # 监视线程让长时间生成也能响应 ComfyUI 的取消操作;
        # 收尾放 finally: 中断/异常路径同样需要 force_offload 释放显存与 hybrid 重置
        try:
            with InterruptWatcher(LLAMA_CPP_STORAGE.llm, log_prefix=node_log_prefix(self.LOG_NAME)) as watcher:
                out = runner(messages, user_content, seed, params, extract_text, watcher)
            if watcher.interrupted:
                # abort_event 使生成提前返回了截断结果, 丢弃并走标准中断流程
                raise mm.InterruptProcessingException()
        finally:
            if force_offload:
                LLAMA_CPP_STORAGE.clean()
            elif LLAMA_CPP_STORAGE.llm is not None and is_hybrid_arch(LLAMA_CPP_STORAGE.llm):
                # 真 hybrid/recurrent 架构(Qwen3.5, LFM2 系等)的线性注意力状态无法
                # 跨请求做前缀复用, 不重置会导致后续请求输出错乱; 按架构判断而非
                # handler 名单, 避免每加一个 hybrid 模型都要维护名单.
                # 重置只在节点执行收尾做一次: image 逐张模式中间的多次请求之间
                # 不重置, 依赖 wheel 内置的 hybrid checkpoint 前缀匹配
                llm = LLAMA_CPP_STORAGE.llm
                llm.n_tokens = 0
                llm._ctx.memory_clear(True)
                if llm._hybrid_cache_mgr is not None:
                    llm._hybrid_cache_mgr.clear()
                logger.debug(node_log_prefix(self.LOG_NAME) + _LOGS["hybrid_reset"])

        return (out,)


class llama_cpp_media_instruct_base(llama_cpp_instruct_base):
    MODEL_TYPE = "LLAMACPPVLM"
    REQUIRE_USER_TEXT = False

    @staticmethod
    def require_mmproj(kind):
        if not getattr(LLAMA_CPP_STORAGE.chat_handler, "mmproj_path", None):
            raise ValueError(_ERRORS["mmproj_not_configured"].format(kind=kind))
