"""System Prompt Preset 节点, 输出图像/视频生成模型的中文提示词增强 system prompt."""

from .system_prompt_presets import PRESETS


class system_prompt_preset:
    CATEGORY = "llama-cpp-vulkan"
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        # "None" 占位在首位(即默认值), 与 model / mmproj / chat_handler 下拉框的
        # 显式选择惯例一致; 只在节点层特判, 不进 PRESETS 模板池
        return {
            "required": {
                "preset": (["None"] + list(PRESETS),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)

    def main(self, preset):
        # 空字符串下游 Instruct 本就不注入 system 消息, 语义自然衔接
        if preset == "None":
            return ("",)
        # 经连线传入的失配预设名不暴露裸 KeyError (widget 常量有 combo 前置校验)
        try:
            return (PRESETS[preset],)
        except KeyError:
            raise ValueError(f'Unknown preset: "{preset}". Re-select a preset from the dropdown.') from None
