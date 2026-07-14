"""llama.cpp video Instruct 节点, 视频帧序列推理.

frames 输入为 IMAGE 帧批次(ComfyUI 生态的视频通行形态, VHS/视频生成模型均输出帧批次).
按 max_frames 均匀采样 -> 缩放到 max_size -> 全部帧并入一条消息,
并在 system prompt 前注入"连续视频"语义提示.
"""

import numpy as np

from .....core.instruct import llama_cpp_media_instruct_base
from ..encoding import image_content_item, scale_image, tensor_to_uint8


def sample_frame_indices(total_frames, max_frames):
    """均匀采样的帧索引; clamp 到实际帧数, 避免 linspace 重复采样同一帧浪费上下文."""
    n_frames = min(max_frames, total_frames)
    return np.linspace(0, total_frames - 1, n_frames, dtype=int)


class llama_cpp_video_instruct(llama_cpp_media_instruct_base):
    MEDIA_WORD = "视频"
    MODALITY = "video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": (cls.MODEL_TYPE,),
                "frames": ("IMAGE", {"tooltip": "IMAGE 帧批次形式的视频帧(如 VHS Load Video 或视频模型 VAE Decode 的输出)."}),
                **cls.seed_input(),
                **cls.prompt_inputs(),
                "max_frames": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1, "tooltip": "从输入帧中均匀采样的帧数上限."}),
                "max_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 16384,
                        "step": 64,
                        "tooltip": "采样帧的最大边长.\n仅在发送多帧时生效, 单帧保持原分辨率.",
                    },
                ),
                **cls.runtime_inputs(),
            },
            "optional": cls.optional_inputs(),
        }

    def process(
        self,
        vlm_model,
        frames,
        seed,
        preset_prompt,
        custom_prompt,
        system_prompt,
        max_frames,
        max_size,
        strip_thinking,
        force_offload,
        parameters=None,
        queue_handler=None,
    ):
        # queue_handler 仅靠连线本身控制执行顺序, 值无用途; del 显式声明有意不使用
        del queue_handler
        # 注入句与用户 system_prompt 之间加换行分隔, 避免两段指令粘连成一句
        video_hint = "请将输入的图像序列视为一段连续的视频, 而不是彼此独立的静态帧."
        system_prompt = (video_hint + "\n" + system_prompt) if system_prompt.strip() else video_hint

        def runner(messages, user_content, seed, params, extract_text, watcher):
            self.require_mmproj("Video")
            sampled = [frames[i] for i in sample_frame_indices(len(frames), max_frames)]

            for frame in sampled:
                if len(sampled) > 1:
                    user_content.append(image_content_item(scale_image(frame, max_size)))
                else:
                    user_content.append(image_content_item(tensor_to_uint8(frame)))
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
