"""Split Instruct Output 节点, 把 image Instruct 逐张模式的拼接输出拆回字符串列表.

按 "====== Image N ======" 分隔行拆分, 输出 OUTPUT_IS_LIST 的 STRING 列表,
供第三方列表语义节点消费(逐条 map 执行或 INPUT_IS_LIST 整表接收);
无分隔行时输出单元素列表, 普通文本 / 单图结果可安全通过.
"""

from ...shared.text_utils import split_image_results


class split_instruct_output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, text):
        return (split_image_results(text),)
