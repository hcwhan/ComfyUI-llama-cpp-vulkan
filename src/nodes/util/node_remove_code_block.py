"""Unpack Code Block 节点, 去除 LLM 输出首尾的代码围栏标记."""

from ...shared.text_utils import strip_code_fence


class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, input):
        return (strip_code_fence(input),)
