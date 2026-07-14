"""llama.cpp text Instruct 节点, 纯文本推理.

只接受 llm Model Loader 的 LLAMACPPLLM 配置,
预设模板的 @@@ 占位符按文生图语境替换为 "图像"(如 prompt 改写预设).
"""

from ....core.instruct import llama_cpp_instruct_base


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
        strip_thinking,
        force_offload,
        parameters=None,
        queue_handler=None,
    ):
        def runner(messages, user_content, seed, params, extract_text, watcher):
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(llm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
