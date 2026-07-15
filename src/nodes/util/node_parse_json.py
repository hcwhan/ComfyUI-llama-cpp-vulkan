"""Parse JSON 节点, 解析 JSON 字符串, 按点分 key 下钻取值, 输出五种类型."""

import json

from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.lang import LANG
from ...shared.text_utils import get_nested_value, parse_json
from ...shared.types import any_type

_NODE = LANG["nodes"]["util"]["parse_json"]


# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    CATEGORY = _CATEGORY
    FUNCTION = "process"
    DESCRIPTION = _NODE["description"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
                "key": (
                    "STRING",
                    {"default": "", "tooltip": _NODE["tooltips"]["key"]},
                ),
            },
            "optional": {
                "default": ("STRING",),
            },
        }

    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")

    def process(self, input, key, default=None):
        if not key.strip():
            raise ValueError(_NODE["errors"]["key_empty"])

        # optional STRING 在常规 UI 下渲染为文本 widget, 未填写时收到空串而非
        # None (None 仅在 widget 转输入端口未连线或 API 直呼时可达);
        # 空串归一为 None, 使两种连线形态的 any 输出一致 (与 DESCRIPTION 对齐)
        if default == "":
            default = None

        # parse_json 统一顶层报错(含代码围栏剥离), 嵌套字符串由 get_nested_value 容错
        val = get_nested_value(parse_json(input), key, default)

        # 转换失败时回退类型零值, 保证输出与声明的 INT/FLOAT 类型一致,
        # 不能把原始值(可能是 str/dict)透传给下游节点.
        # int 不经 float 中转: 超过 2^53 的大整数(如雪花 ID)会在 float 中丢精度.
        # OverflowError: json.loads 接受 Infinity 字面量返回 float('inf')
        try:
            integer = int(val)
        except (TypeError, ValueError, OverflowError):
            try:
                integer = int(float(val))  # "1.5" 之类的数字字符串
            except (TypeError, ValueError, OverflowError):
                integer = 0
        try:
            number = float(val)
        except (TypeError, ValueError, OverflowError):
            number = 0.0

        if isinstance(val, bool):
            boolean = val
        elif isinstance(val, (int, float)):
            # 数字按非零判定, 对齐常见 truthy 直觉(JSON 中 1/0 常当布尔用)
            boolean = val != 0
        else:
            boolean = str(val).strip().lower() == "true"

        # dict/list 输出合法 JSON 文本而非 Python repr, 下游可再接 Parse JSON;
        # key 未命中且未连 default 时 val 为 None, string 输出回退空串,
        # 避免字面 "None" 被下游当作有效文本拼进 prompt
        if isinstance(val, (dict, list)):
            string = json.dumps(val, ensure_ascii=False)
        elif val is None:
            string = ""
        else:
            string = str(val)

        return (val, string, integer, number, boolean)
