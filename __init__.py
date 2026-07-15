"""插件入口, 向 ComfyUI 导出节点注册表与前端扩展目录."""

import logging

try:
    from .src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # 依赖缺失 (未按 requirements.txt 安装) 或平台不受支持 (如 macOS) 时,
    # ComfyUI 日志只有裸 traceback, 先给出安装指引再原样抛出.
    # i18n 层可能同在失败的导入链上, 此处例外地使用硬编码英文文案
    logging.getLogger("llama-cpp-vulkan").error(
        "[llama-cpp-vulkan] import failed: install the pinned llama-cpp-python Vulkan wheel via requirements.txt (Windows / Linux x86_64 only)."
    )
    raise

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
