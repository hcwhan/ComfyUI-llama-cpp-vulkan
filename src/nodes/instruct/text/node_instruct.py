"""llama.cpp text Instruct 节点, 纯文本推理.

只接受 LLM Model Loader 的 LLAMACPPLLM 配置,
预设模板的 @@@ 占位符按文生图语境替换为 "图像"(如 prompt 改写预设).
"""

from ....core.instruct import llama_cpp_instruct_base, think_open_preinjected
from ....core.storage import LLAMA_CPP_STORAGE
from ....i18n.lang import LANG
from ....shared.logger import logger, node_log_prefix

_LOGS = LANG["logs"]["text_instruct"]


class llama_cpp_text_instruct(llama_cpp_instruct_base):
    MODEL_TYPE = "LLAMACPPLLM"
    MEDIA_WORD = "图像"
    MODALITY = "text"
    LOG_NAME = "Text Instruct"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (cls.MODEL_TYPE,),
                **cls.seed_input(),
                **cls.prompt_inputs(),
                "allow_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": LANG["nodes"]["instruct"]["text"]["tooltips"]["allow_thinking"],
                    },
                ),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def process(
        self,
        llm_model,
        seed,
        preset_prompt,
        custom_prompt,
        system_prompt,
        allow_thinking,
        strip_thinking,
        force_offload,
        parameters=None,
        queue_handler=None,
    ):
        # queue_handler 仅靠连线本身控制执行顺序, 值无用途; del 显式声明有意不使用
        del queue_handler

        def runner(messages, user_content, seed, params, extract_text, watcher):
            # 折算: False -> reasoning_budget=0 (思考块开启即强制闭合), True -> -1
            # (采样器整体停用, 不干预). False 时按模型的思考形态分四种场景:
            # 1. 模型自己生成 <think> 开标签 (Qwen3 经典等): 采样器等到生成的
            #    开标签立即强制闭合成 <think></think> 空块, 正常抑制;
            # 2. 模板预注入 <think> 到 generation prompt (Qwen3.5 等): 采样器等不到
            #    生成的开标签, 由 think_open_preinjected 渲染探测识别, 改传
            #    reasoning_start_in_prompt=True 使其跳过等开标签直接强制闭合;
            #    探测失败按未预注入处理, 此时思考块照常生成, 仍由 strip_thinking 剥离;
            # 3. 非思考模型: 采样器在 reasoning_start 安全窗 (默认 32 token) 内
            #    等不到开标签即静默退场, 无害;
            # 4. 思考开标签非 <think> 的模型 (如 <thought>, <seed:think>): 同 3 经
            #    安全窗静默失效, 且其思考块不属 strip_thinking 的剥离形态, 按不支持处理.
            # (params 是 _run 的防御性副本, 可安全写入)
            params["reasoning_budget"] = -1 if allow_thinking else 0
            params["reasoning_start_in_prompt"] = not allow_thinking and think_open_preinjected(LLAMA_CPP_STORAGE.llm)
            logger.info(
                node_log_prefix(self.LOG_NAME)
                + _LOGS["allow_thinking"].format(
                    allow_thinking=allow_thinking,
                    reasoning_budget=params["reasoning_budget"],
                    reasoning_start_in_prompt=params["reasoning_start_in_prompt"],
                )
            )
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(llm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
