"""手动卸载节点, 释放模型与 chat handler 资源(any 透传, 可串接任意连线).

卸载是 process() 的副作用, 受 ComfyUI 结果缓存约束: 上游输出未变化时
节点被缓存跳过, 不会重复卸载(设计权衡, 避免强制执行导致下游每次排队
全链路重算). 重跑未改动的工作流时若显存中残留其他工作流加载的模型,
可用前端 Free 按钮(已挂 unload_all_models 钩子)或 Instruct 节点的
force_offload 开关兜底.
"""

from ...shared.logger import logger
from ...shared.types import any_type
from ...core.storage import LLAMA_CPP_STORAGE


class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type, {"tooltip": "any 透传端口, 串接在需要先卸载模型的连线上.\n注意: 仅在上游输出变化时执行 (ComfyUI 缓存语义),\n重跑未改动的工作流不会重复卸载."})}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, any):
        logger.info("[llama-cpp-vulkan] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)
