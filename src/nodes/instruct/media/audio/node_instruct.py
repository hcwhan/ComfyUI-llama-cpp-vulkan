"""llama.cpp audio Instruct 节点, 音频推理(ASR/omni, 如 Qwen3-ASR).

音频以 16-bit 单声道 WAV 注入 input_audio 内容项, 重采样由 llama.cpp 的
mtmd 解码端完成. 音频是否被 mmproj 支持由 llama-cpp-python 侧校验.
"""

from .....core.instruct import llama_cpp_media_instruct_base
from .....i18n.lang import LANG
from .....shared.encoding import audio2base64
from .....shared.logger import logger, node_log_prefix

_LOGS = LANG["logs"]["audio_instruct"]


class llama_cpp_audio_instruct(llama_cpp_media_instruct_base):
    MODALITY = "audio"
    LOG_NAME = "Audio Instruct"
    MEDIA_WORD = "音频"

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
            # 时长按原始波形计, 重采样在 llama.cpp 的 mtmd 解码端, 不改变时长
            sample_rate = int(audio["sample_rate"])
            logger.info(
                node_log_prefix(self.LOG_NAME)
                + _LOGS["input"].format(duration=audio["waveform"].shape[-1] / sample_rate, sample_rate=sample_rate)
            )
            user_content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio2base64(audio), "format": "wav"},
                }
            )
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
