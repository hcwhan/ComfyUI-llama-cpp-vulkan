"""Split Instruct Output 节点, 把 image Instruct 逐张模式的拼接输出拆回字符串列表.

按 "======== Image N ========" 前缀行拆分, 输出 OUTPUT_IS_LIST 的 STRING 列表,
供第三方列表语义节点消费(逐条 map 执行或 INPUT_IS_LIST 整表接收);
无前缀行时输出单元素列表, 普通文本 / 单图结果可安全通过.
"""

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix
from ...shared.text_utils import split_image_results

_LOGS = LANG["logs"]["util"]
_PREFIX = node_log_prefix("Split Instruct Output")


class split_instruct_output:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)

    def process(self, text):
        parts = split_image_results(text)
        logger.info(_PREFIX + _LOGS["split_output"].format(count=len(parts)))
        return (parts,)
