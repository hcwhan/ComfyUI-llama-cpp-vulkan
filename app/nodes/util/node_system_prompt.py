"""System Prompt Preset 节点, 输出图像/视频生成模型的中文提示词增强 system prompt."""

from .system_prompt_presets import PRESETS


class system_prompt_preset:
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
