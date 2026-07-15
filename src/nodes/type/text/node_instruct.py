"""llama.cpp text Instruct 节点, 纯文本推理.

只接受 llm Model Loader 的 LLAMACPPLLM 配置,
预设模板的 @@@ 占位符按文生图语境替换为 "图像"(如 prompt 改写预设).
"""

from ....core.instruct import llama_cpp_instruct_base
from ....i18n.lang import LANG


class llama_cpp_text_instruct(llama_cpp_instruct_base):
    MODEL_TYPE = "LLAMACPPLLM"
    MEDIA_WORD = "图像"
    MODALITY = "text"

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
            # False -> reasoning_budget=0 (思考块开启即强制闭合), True -> -1 (不干预).
            # 非思考模型由 wheel 采样器的 reasoning_start 安全窗自动失效, 无害;
            # 模板预注入 <think> 的形态采样器等不到生成的开标签而无法抑制,
            # 残留思考块仍由 strip_thinking 剥离 (params 是 _run 的防御性副本, 可安全写入)
            params["reasoning_budget"] = -1 if allow_thinking else 0
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(llm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
