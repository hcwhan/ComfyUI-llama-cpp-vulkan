from ..support.prompt_enhancer_preset import PRESETS
from .shared import any_type, get_nested_value, parse_json, strip_code_fence


# from: https://github.com/crystian/ComfyUI-Crystools
class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "key": ("STRING",),
                "default": ("STRING",),
            },
        }

    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, input, key=None, default=None):
        if not key:
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


class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "label": ("STRING",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, input, label=""):
        return (strip_code_fence(input, label),)


class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (list(PRESETS),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vulkan"

    def main(self, preset):
        return (PRESETS[preset],)
