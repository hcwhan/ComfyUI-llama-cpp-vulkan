"""插件入口, 向 ComfyUI 导出节点注册表与前端扩展目录, 挂接前端语言上报路由."""

import logging

try:
    from .src.core import locale_sync  # noqa: F401  (import 即注册路由)
    from .src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    # 依赖缺失 (未按 requirements.txt 安装), 平台不受支持 (如 macOS), 以及
    # wheel 已装但原生库加载失败 (缺 VC++ 运行库 / glibc 过旧 / 文件损坏,
    # 抛 RuntimeError; C 符号缺失抛 AttributeError, 均非 ImportError 子类,
    # 故按 Exception 捕获) 时, ComfyUI 日志只有裸 traceback,
    # 先给出安装指引再原样抛出.
    # i18n 层可能同在失败的导入链上, 此处例外地使用硬编码英文文案
    logging.getLogger("llama-cpp-vulkan").error(
        "[llama-cpp-vulkan] import failed (missing dependency or native library load failure): install (or reinstall) the pinned llama-cpp-python Vulkan wheel via requirements.txt (Windows / Linux x86_64 only)."
    )
    raise

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
