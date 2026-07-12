"""Parse JSON 节点, 解析 JSON 字符串, 按点分 key 下钻取值, 输出五种类型."""

from ...shared.types import any_type
from ...shared.text_utils import get_nested_value, parse_json


# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        # key 运行期必填,声明为 required 保持 UI 语义一致;
        # 从 optional 移入不影响旧工作流: input 是 forceInput 无 widget 值,
        # widget 序列化顺序仍为 [key, default]
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"default": "", "tooltip": "点分路径下钻取值, 如 a.b.c"}),
            },
            "optional": {
                "default": ("STRING",),
            },
        }

    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, input, key, default=None):
        if not key.strip():
            raise ValueError("Key cannot be empty!")

        # parse_json 统一顶层报错(含代码围栏剥离),嵌套字符串由 get_nested_value 容错
        val = get_nested_value(parse_json(input), key, default)

        # 转换失败时回退类型零值,保证输出与声明的 INT/FLOAT 类型一致,
        # 不能把原始值(可能是 str/dict)透传给下游节点
        try:
            number = float(val)
            integer = int(number)
        except (TypeError, ValueError, OverflowError):
            number, integer = 0.0, 0

        if isinstance(val, bool):
            boolean = val
        else:
            boolean = str(val).strip().lower() == "true"

        return (val, str(val), integer, number, boolean)
