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
        # 经连线传入的失配预设名不暴露裸 KeyError (widget 常量有 combo 前置校验)
        try:
            return (PRESETS[preset],)
        except KeyError:
            raise ValueError(f'Unknown preset: "{preset}". Re-select a preset from the dropdown.') from None
