"""手动卸载节点, 释放模型与 chat handler 资源(any 透传, 可串接任意连线)."""

from ...shared.logger import logger
from ...shared.types import any_type
from ...core.storage import LLAMA_CPP_STORAGE


class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, any):
        logger.info("[llama-cpp-vulkan] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)
