"""测试专用: 在导入 app 包之前注入 comfy/folder_paths 的最小替身模块.

app 的部分模块在 import 时就访问 ComfyUI 运行时(注册模型目录, monkey-patch
卸载钩子, 取 ProgressBar 类), 测试进程没有 ComfyUI, 用替身满足 import 期依赖.
llama_cpp 使用真实安装(纯函数测试不触发推理).
"""

import sys
import types
import tempfile


def install():
    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")

    class InterruptProcessingException(Exception):
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

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = utils
    sys.modules["folder_paths"] = folder_paths
