"""Unpack Code Block 节点, 去除 LLM 输出首尾的代码围栏标记; 围栏块外有前导/尾随说明时, 回退为提取文本中第一个完整围栏块的内容."""

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.logger import logger, node_log_prefix
from ...shared.text_utils import strip_code_fence

_LOGS = LANG["logs"]["util"]
_PREFIX = node_log_prefix("Unpack Code Block")


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
        output = strip_code_fence(input)
        logger.info(_PREFIX + _LOGS["remove_code_block"].format(before=len(input), after=len(output)))
        return (output,)
