"""llama.cpp video Instruct 节点, 视频帧序列推理.

frames 输入为 IMAGE 帧批次(ComfyUI 生态的视频通行形态, VHS/视频生成模型均输出帧批次).
按 max_frames 均匀采样 -> 缩放到 max_size(单帧不缩放, 保持原分辨率) -> 全部帧并入一条消息,
并在 system prompt 前注入"连续视频"语义提示.
"""

import numpy as np

from .....core.instruct import llama_cpp_media_instruct_base
from .....i18n.lang import LANG
from .....shared.encoding import image_content_item, scale_image, tensor_to_uint8
from .....shared.logger import logger, node_log_prefix

_TIPS = LANG["nodes"]["instruct"]["video"]["tooltips"]
_LOGS = LANG["logs"]["video_instruct"]


def sample_frame_indices(total_frames, max_frames):
    """均匀采样的帧索引; clamp 到实际帧数, 避免 linspace 重复采样同一帧浪费上下文."""
    n_frames = min(max_frames, total_frames)
    return np.linspace(0, total_frames - 1, n_frames, dtype=int)


class llama_cpp_video_instruct(llama_cpp_media_instruct_base):
    MODALITY = "video"
    LOG_NAME = "Video Instruct"
    MEDIA_WORD = "视频"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vlm_model": (cls.MODEL_TYPE,),
                "frames": ("IMAGE", {"tooltip": _TIPS["frames"]}),
                **cls.seed_input(),
                **cls.prompt_inputs(),
                "max_frames": ("INT", {"default": 30, "min": 2, "max": 1024, "step": 1, "tooltip": _TIPS["max_frames"]}),
                "max_size": (
                    "INT",
                    {
                        "default": 256,
                        "min": 128,
                        "max": 16384,
                        "step": 64,
                        "tooltip": _TIPS["max_size"],
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

        video_hint = "请将输入的图像序列视为一段连续的视频, 而不是彼此独立的静态帧."
        system_prompt = (video_hint + "\n" + system_prompt) if system_prompt.strip() else video_hint

        def runner(messages, user_content, seed, params, extract_text, watcher):
            self.require_mmproj("Video")
            sampled = [frames[i] for i in sample_frame_indices(len(frames), max_frames)]
            logger.info(
                node_log_prefix(self.LOG_NAME) + _LOGS["sampling"].format(total=len(frames), sampled=len(sampled), max_size=max_size)
            )

            for frame in sampled:
                if len(sampled) > 1:
                    user_content.append(image_content_item(scale_image(frame, max_size)))
                else:
                    user_content.append(image_content_item(tensor_to_uint8(frame)))
            return self._single_completion(messages, user_content, seed, params, extract_text)

        return self._run(vlm_model, seed, preset_prompt, custom_prompt, system_prompt, strip_thinking, force_offload, parameters, runner)
