"""测试专用: 在导入 src 包之前注入 comfy/folder_paths/server 的最小替身模块.

src 的部分模块在 import 时就访问 ComfyUI 运行时(注册模型目录, monkey-patch
卸载钩子, 取 ProgressBar 类), 测试进程没有 ComfyUI, 用替身满足 import 期依赖.
llama_cpp 使用真实安装(纯函数测试不触发推理).
"""

import os
import sys
import tempfile
import types


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

    # 指向一个不存在的目录 (不创建): i18n 的语言自动检测在测试进程中
    # 确定性地走 "设置文件缺失" 分支, 回退默认英语
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
