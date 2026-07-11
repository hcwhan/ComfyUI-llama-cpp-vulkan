from ..support.prompt_enhancer_preset import PRESETS
from .shared import any_type, get_nested_value


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

        val = get_nested_value(input.strip().removeprefix("```json").removesuffix("```"), key, default)

        def coerce(fn):
            try:
                return fn(val)
            except Exception:
                return val

        if isinstance(val, bool):
            boolean = val
        else:
            boolean = str(val).strip().lower() == "true"

        return (val, coerce(str), coerce(int), coerce(float), boolean)


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
        return (input.strip().removeprefix(f"```{label}").removesuffix("```"),)


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
