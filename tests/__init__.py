"""单元测试包(标准库 unittest, 不引入 pytest 依赖).

运行方式(仓库根目录): python -m unittest discover -s tests -t . -v
使用 ComfyUI 的嵌入式 Python 运行时, torch/llama_cpp 为真实安装,
comfy/folder_paths 由 comfy_stubs 注入最小替身.

插件的 app 是隐式命名空间包, 而嵌入式 Python 的 ._pth 使 ComfyUI 主程序
目录先于测试路径, 其常规包 app 会胜过命名空间包; 这里显式把插件的 app
目录注册进 sys.modules, 保证测试导入的是本仓库的代码.
"""

import sys
import importlib.machinery
import importlib.util
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent / "app"

if "app" not in sys.modules:
    _spec = importlib.machinery.ModuleSpec("app", None, is_package=True)
    _spec.submodule_search_locations = [str(_APP_DIR)]
    sys.modules["app"] = importlib.util.module_from_spec(_spec)
