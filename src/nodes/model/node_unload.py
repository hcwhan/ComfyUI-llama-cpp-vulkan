"""手动卸载节点, 释放模型与 chat handler 资源(any 透传, 可串接任意连线).

卸载是 process() 的副作用, 受 ComfyUI 结果缓存约束: 上游输出未变化时
节点被缓存跳过, 不会重复卸载(设计权衡, 避免强制执行导致下游每次排队
全链路重算). 重跑未改动的工作流时若显存中残留其他工作流加载的模型,
可用前端 Free 按钮(已挂 unload_all_models 钩子)或 Instruct 节点的
force_offload 开关兜底.
"""

from ...core.storage import LLAMA_CPP_STORAGE
from ...i18n.common_static import CATEGORY as _CATEGORY
from ...i18n.common_static import LOG_PREFIX
from ...i18n.lang import LANG
from ...shared.logger import logger
from ...shared.types import any_type


class llama_cpp_unload_model:
    CATEGORY = _CATEGORY
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (
                    any_type,
                    {"tooltip": LANG["nodes"]["model"]["unload"]["tooltips"]["any"]},
                )
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)

    def process(self, any):
        # 空载时不打 "Unloading" 日志(避免误导排查); clean() 幂等, 仍无条件执行兜底
        if LLAMA_CPP_STORAGE.llm is not None:
            logger.info(LOG_PREFIX + LANG["logs"]["unload"]["unloading"])
        LLAMA_CPP_STORAGE.clean()
        return (any,)
