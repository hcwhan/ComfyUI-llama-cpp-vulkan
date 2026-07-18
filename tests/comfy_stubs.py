"""测试专用: 在导入 src 包之前注入 comfy/folder_paths/server 的最小替身模块.

src 的部分模块在 import 时就访问 ComfyUI 运行时(注册模型目录, monkey-patch
卸载钩子, import comfy.utils), 测试进程没有 ComfyUI, 用替身满足 import 期依赖.
另将 i18n 的插件根 settings.json 路径重定向出仓库根, 隔离语言解析的持久化
状态(理由见 install 内注释). llama_cpp 使用真实安装(纯函数测试不触发推理).
"""

import os
import sys
import tempfile
import types
from pathlib import Path


def install():
    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")

    # 与真实 ComfyUI 保持一致继承 BaseException(而非 Exception):
    # 否则替身会掩盖生产代码用 except Exception 误吞中断一类的问题
    class InterruptProcessingException(BaseException):
        pass

    mm.InterruptProcessingException = InterruptProcessingException
    mm.processing_interrupted = lambda: False
    mm.unload_all_models = lambda *args, **kwargs: None
    mm.free_memory = lambda *args, **kwargs: []
    mm.get_torch_device = lambda: None

    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.current = 0

        def update(self, value):
            self.current += value

        def update_absolute(self, value, total=None, preview=None):
            if total is not None:
                self.total = total
            self.current = value

    utils.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = utils

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = tempfile.gettempdir()
    folder_paths.folder_names_and_paths = {}

    def add_model_folder_path(key, path, is_default=False):
        folder_paths.folder_names_and_paths.setdefault(key, ([path], set()))

    folder_paths.add_model_folder_path = add_model_folder_path
    folder_paths.get_filename_list = lambda key: []
    folder_paths.get_full_path = lambda key, filename: None

    # 语言解析第 1 级 (Comfy.Locale) 的隔离: 指向一个不存在的目录 (不创建),
    # 确定性地走 "设置文件缺失" 分支; 第 2 级的隔离见 install 末尾的
    # settings.json 重定向, 两级齐备才保证 import 期 LANG 确定性回退默认英语
    _user_dir = os.path.join(tempfile.gettempdir(), "comfyui-llama-cpp-vulkan-tests", "user")
    folder_paths.get_user_directory = lambda: _user_dir

    server = types.ModuleType("server")

    class _RouteTable:
        """PromptServer.instance.routes 替身: 只收集注册项, 供测试直接驱动 handler."""

        def __init__(self):
            self.registered = []

        def post(self, path):
            def decorator(handler):
                self.registered.append(("POST", path, handler))
                return handler

            return decorator

    class PromptServer:
        pass

    PromptServer.instance = PromptServer()
    PromptServer.instance.routes = _RouteTable()
    server.PromptServer = PromptServer

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = utils
    sys.modules["folder_paths"] = folder_paths
    sys.modules["server"] = server

    # 语言解析第 2 级 (插件根 settings.json) 的隔离: 该文件是 gitignore 的
    # 运行时产物, junction 部署 (custom_nodes 直指本仓库) 下正常使用 ComfyUI
    # 后会真实存在于仓库根. 若不重定向: (a) 测试进程 import 期 LANG 会按其
    # frontend_locale 解析而非确定性回退默认英语; (b) lang import 期
    # read_comfy_locale() 以 set_language_setting("comfy_locale", None) 收尾,
    # 跑一次单测就会改写生产运行时文件. 预导入 locale_settings (须在替身
    # 注册之后: 其 import 期有 import folder_paths; 且早于 lang import,
    # 时序可行) 并把 _SETTINGS_PATH 指向临时目录下不存在的文件 (不创建),
    # 确定性走 "文件缺失" 分支; 单测内的针对性 patch 仍按用例自行覆盖.
    from src.i18n import locale_settings

    locale_settings._SETTINGS_PATH = Path(tempfile.gettempdir()) / "comfyui-llama-cpp-vulkan-tests" / "settings.json"
