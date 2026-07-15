"""Unpack Code Block 节点, 去除 LLM 输出首尾的代码围栏标记."""

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...shared.text_utils import strip_code_fence


class remove_code_block:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)

    def process(self, input):
        return (strip_code_fence(input),)
