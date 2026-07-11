"""Vulkan GPU 设备检测与选择。

通过 ggml C API (ctypes) 直接枚举后端设备,区分独显 (GPU) 和核显 (IGPU),
是独立于 PyTorch/CUDA 的 Vulkan 推理路径。

设备枚举在模块 import 时同步执行(约几百 ms),属有意设计:
UI 下拉框需要在启动期确定设备列表。
"""
import ctypes
from pathlib import Path

import llama_cpp.llama_cpp as _llama_cpp_lib
from llama_cpp._ggml import (
    libggml_base,
    ggml_backend_dev_count,
    ggml_backend_dev_get,
    ggml_backend_load_all_from_path,
)

libggml_base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_name.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_description.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_type.restype = ctypes.c_int32

_GGML_BACKEND_DEVICE_TYPE_GPU = 1
_GGML_BACKEND_DEVICE_TYPE_IGPU = 2
_DEV_TYPE_NAMES = {1: "GPU", 2: "IGPU"}

AUTO_LABEL = "Auto (独显优先)"

SPLIT_MODE_NONE = _llama_cpp_lib.llama_split_mode.LLAMA_SPLIT_MODE_NONE
SPLIT_MODE_LAYER = _llama_cpp_lib.llama_split_mode.LLAMA_SPLIT_MODE_LAYER


def _detect_gpu_devices():
    try:
        lib_dir = Path(_llama_cpp_lib.__file__).resolve().parent / "lib"
        if not lib_dir.exists():
            return []
        ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))

        devices = []
        for i in range(ggml_backend_dev_count()):
            dev = ggml_backend_dev_get(i)
            dev_type = libggml_base.ggml_backend_dev_type(dev)
            if dev_type in (_GGML_BACKEND_DEVICE_TYPE_GPU, _GGML_BACKEND_DEVICE_TYPE_IGPU):
                name = libggml_base.ggml_backend_dev_name(dev).decode("utf-8", errors="replace")
                desc = libggml_base.ggml_backend_dev_description(dev).decode("utf-8", errors="replace").strip()
                type_name = _DEV_TYPE_NAMES.get(dev_type, "GPU")
                devices.append({"name": name, "desc": desc, "type": type_name})
        return devices
    except Exception as e:
        print(f"[llama-cpp-vulkan] WARNING: GPU detection failed: {e}")
        return []


_gpu_devices = _detect_gpu_devices()

if _gpu_devices:
    _summary = ", ".join(f"{d['name']} ({d['desc']}) [{d['type']}]" for d in _gpu_devices)
    print(f"[llama-cpp-vulkan] Detected {len(_gpu_devices)} GPU device(s): {_summary}")
else:
    print("[llama-cpp-vulkan] WARNING: No GPU devices detected, running on CPU only")


def _selectable_devices():
    """按 llama.cpp 收集 model->devices 的规则,返回 main_gpu 实际可选的设备。

    llama.cpp 只把独显 (type==GPU) 按枚举顺序加入设备列表;
    仅当系统没有任何独显时,才加入第一个核显 (IGPU)。
    其余设备无法通过 main_gpu 参数选中,因此不在下拉框中展示。
    """
    dgpus = [d for d in _gpu_devices if d["type"] == "GPU"]
    if dgpus:
        return dgpus
    return [d for d in _gpu_devices if d["type"] == "IGPU"][:1]


def _device_label(dev):
    return f"{dev['name']} - {dev['desc']} [{dev['type']}]"


def resolve_device_selection(gpu_device):
    """把下拉框选项翻译为 (main_gpu, split_mode)。

    main_gpu 仅在 split_mode=NONE 时生效,且索引是相对 llama.cpp 的
    model->devices 列表(即 _selectable_devices 的顺序),不是 ggml 全局设备序号。
    Auto 保持 llama.cpp 默认行为:LAYER 模式,独显优先,多独显按层切分。
    """
    if gpu_device != AUTO_LABEL:
        for i, dev in enumerate(_selectable_devices()):
            if _device_label(dev) == gpu_device:
                return i, SPLIT_MODE_NONE
        print(f"[llama-cpp-vulkan] WARNING: device '{gpu_device}' is not selectable, falling back to Auto")
    return 0, SPLIT_MODE_LAYER


gpu_device_choices = [AUTO_LABEL] + [_device_label(d) for d in _selectable_devices()]


def print_backend_summary(main_gpu, split_mode):
    selectable = _selectable_devices()
    if not selectable:
        print("[llama-cpp-vulkan] WARNING: No GPU backend detected, running on CPU only")
        return
    if split_mode == SPLIT_MODE_LAYER and len(selectable) > 1:
        names = ", ".join(d["name"] for d in selectable)
        print(f"[llama-cpp-vulkan] Active GPUs (layer split): {names}")
    else:
        active = selectable[main_gpu] if main_gpu < len(selectable) else selectable[0]
        print(f"[llama-cpp-vulkan] Active GPU: {active['name']} ({active['desc']}) [{active['type']}]")
