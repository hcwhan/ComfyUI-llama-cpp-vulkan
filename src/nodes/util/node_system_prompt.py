"""System Prompt Preset 节点, 输出图像/视频生成模型的中文提示词增强 system prompt."""

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.common_static import NONE_OPTION
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix
from .system_prompt_presets import PRESETS

_LOGS = LANG["logs"]["util"]
_PREFIX = node_log_prefix("System Prompt Preset")


class system_prompt_preset:
    CATEGORY = _CATEGORY
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(cls):
        # "None" 占位在首位(即默认值), 与 model / mmproj / chat_handler 下拉框的
        # 显式选择惯例一致; 只在节点层特判, 不进 PRESETS 模板池
        return {
            "required": {
                "preset": ([NONE_OPTION] + list(PRESETS),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)

    def main(self, preset):
        # 空字符串下游 Instruct 本就不注入 system 消息, 语义自然衔接
        if preset == NONE_OPTION:
            return ("",)
        # 经连线传入的失配预设名不暴露裸 KeyError (widget 常量有 combo 前置校验)
        try:
            text = PRESETS[preset]
        except KeyError:
            raise ValueError(LANG["nodes"]["util"]["system_prompt_preset"]["errors"]["unknown_preset"].format(preset=preset)) from None
        logger.info(_PREFIX + _LOGS["system_prompt"].format(preset=preset, chars=len(text)))
        return (text,)
