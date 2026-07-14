"""Parse JSON 节点, 解析 JSON 字符串, 按点分 key 下钻取值, 输出五种类型."""

import json

from ...shared.text_utils import get_nested_value, parse_json
from ...shared.types import any_type


# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    CATEGORY = "llama-cpp-vulkan"
    FUNCTION = "process"
    DESCRIPTION = (
        "解析 JSON 字符串并按点分 key 取值, 同一个值以五种类型输出.\n"
        "转换规则: string 对 dict/list 输出合法 JSON 文本, 其余为 str() 结果;\n"
        "int/float 转换失败时回退 0 / 0.0; boolean 对数字取非零判定,\n"
        '对文本仅 "true" (忽略大小写) 为真.\n'
        'key 未命中且未连 default 时输出 (None, "", 0, 0.0, False).'
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
                "key": (
                    "STRING",
                    {"default": "", "tooltip": "点分路径下钻取值, 如 a.b.c\n数组用数字下标, 如 items.0.label (负数从尾部取)"},
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
            raise ValueError("Key cannot be empty!")

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
