"""单元测试包(标准库 unittest, 不引入 pytest 依赖).

运行方式(仓库根目录): python -m unittest discover -s tests -t . -v
使用 ComfyUI 的嵌入式 Python 运行时, torch/llama_cpp 为真实安装,
comfy/folder_paths 由 comfy_stubs 注入最小替身.

插件的 src 是隐式命名空间包, 而嵌入式 Python 的 ._pth 把 ComfyUI 主程序
目录固定加入 sys.path; 路径上任何位置若出现同名常规包 src, 都会使命名空间包
作废(常规包优先于命名空间包, 与路径顺序无关); 这里显式把插件的 src 目录
注册进 sys.modules, 保证测试导入的是本仓库的代码.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if "src" not in sys.modules:
    _spec = importlib.machinery.ModuleSpec("src", None, is_package=True)
    _spec.submodule_search_locations = [str(_SRC_DIR)]
    sys.modules["src"] = importlib.util.module_from_spec(_spec)
