"""llama.cpp audio Instruct 节点, 音频推理(ASR/omni, 如 Qwen3-ASR).

音频以 16-bit 单声道 WAV 注入 input_audio 内容项, 重采样由 llama.cpp 的
mtmd 解码端完成. 音频是否被 mmproj 支持由 llama-cpp-python 侧校验.
"""

from .....core.instruct import llama_cpp_media_instruct_base
from .....i18n.lang import LANG
from ..encoding import audio2base64


class llama_cpp_audio_instruct(llama_cpp_media_instruct_base):
    MEDIA_WORD = "音频"
    MODALITY = "audio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": (cls.MODEL_TYPE,),
                "audio": ("AUDIO", {"tooltip": LANG["nodes"]["instruct"]["audio"]["tooltips"]["audio"]}),
                **cls.seed_input(),
                **cls.prompt_inputs(),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def process(
        self,
        vlm_model,
        audio,
        seed,
        preset_prompt,
        custom_prompt,
        system_prompt,
        strip_thinking,
        force_offload,
        parameters=None,
        queue_handler=None,
    ):
        # queue_handler 仅靠连线本身控制执行顺序, 值无用途; del 显式声明有意不使用
        del queue_handler

        def runner(messages, user_content, seed, params, extract_text, watcher):
            self.require_mmproj("Audio")
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio2base64(audio), "format": "wav"},
                }
            )
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
