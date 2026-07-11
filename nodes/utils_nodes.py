from ..support.prompt_enhancer_preset import *
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
        if isinstance(input, str):
            input = [input]

        result = {"any": {}, "string": {}, "int": {}, "float": {}, "boolean": {}}
        for i, json_str in enumerate(input):
            val = ""
            if key is not None and key != "":
                val = get_nested_value(json_str.strip().removeprefix("```json").removesuffix("```"), key, default)
            else:
                raise ValueError("Key cannot be empty!")

            result["any"][i] = val
            try:
                result["string"][i] = str(val)
            except Exception as e:
                result["string"][i] = val

            try:
                result["int"][i] = int(val)
            except Exception as e:
                result["int"][i] = val

            try:
                result["float"][i] = float(val)
            except Exception as e:
                result["float"][i] = val

            try:
                result["boolean"][i] = val.lower() == "true"
            except Exception as e:
                result["boolean"][i] = val

        if len(result["any"]) == 1:
            result["any"] = result["any"][0]
            result["string"] = result["string"][0]
            result["int"] = result["int"][0]
            result["float"] = result["float"][0]
            result["boolean"] = result["boolean"][0]

        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])


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
        if isinstance(input, str):
            input = [input]

        output = []
        for value in input:
            output.append(value.strip().removeprefix(f"```{label}").removesuffix("```"))
        if len(output) == 1:
            return (output[0],)
        return (output,)


class PromptEnhancerPreset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (["Qwen-Image [EN]", "Qwen-Image [ZH]", "Qwen-Image 2512 [EN]", "Qwen-Image 2512 [ZH]", "Qwen-Image-Edit", "Qwen-Image-Edit 2509", "Qwen-Image-Edit 2511", "Z-Image Turbo", "Flux.2 T2I", "Flux.2 I2I", "Wan T2V [EN]", "Wan T2V [ZH]", "Wan I2V [EN]", "Wan I2V [ZH]", "Wan I2V Full-Auto [EN]", "Wan I2V Full-Auto [ZH]", "Wan FLF2V [EN]", "Wan FLF2V [ZH]"], )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "main"
    CATEGORY = "llama-cpp-vulkan"

    def main(self, preset):
        match preset:
            case "Qwen-Image [EN]":
                return (QWEN_IMAGE_EN,)
            case "Qwen-Image [ZH]":
                return (QWEN_IMAGE_ZH,)
            case "Qwen-Image 2512 [EN]":
                return (QWEN_IMAGE_2512_EN,)
            case "Qwen-Image 2512 [ZH]":
                return (QWEN_IMAGE_2512_ZH,)
            case "Qwen-Image-Edit":
                return (QWEN_IMAGE_EDIT,)
            case "Qwen-Image-Edit 2509":
                return (QWEN_IMAGE_EDIT_2509,)
            case "Qwen-Image-Edit 2511":
                return (QWEN_IMAGE_EDIT_2511,)
            case "Z-Image Turbo":
                return (ZIMAGE_TURBO,)
            case "Flux.2 T2I":
                return (FLUX2_T2I,)
            case "Flux.2 I2I":
                return (FLUX2_I2I,)
            case "Wan T2V [EN]":
                return (WAN_T2V_EN,)
            case "Wan T2V [ZH]":
                return (WAN_T2V_ZH,)
            case "Wan I2V [EN]":
                return (WAN_I2V_EN,)
            case "Wan I2V [ZH]":
                return (WAN_I2V_ZH,)
            case "Wan I2V Full-Auto [EN]":
                return (WAN_I2V_EMPTY_EN,)
            case "Wan I2V Full-Auto [ZH]":
                return (WAN_I2V_EMPTY_ZH,)
            case "Wan FLF2V [EN]":
                return (WAN_FLF2V_EN,)
            case "Wan FLF2V [ZH]":
                return (WAN_FLF2V_ZH,)
            case _:
                raise ValueError(f'Unknown preset: "{preset}"')
