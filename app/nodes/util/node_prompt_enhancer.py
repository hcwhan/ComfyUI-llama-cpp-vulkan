"""Prompt Enhancer Preset 节点, 输出 12 种图像/视频生成模型的中文提示词增强 system prompt."""

from .prompt_enhancer_presets import PRESETS


class prompt_enhancer_preset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (list(PRESETS),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vulkan"

    def main(self, preset):
        return (PRESETS[preset],)
