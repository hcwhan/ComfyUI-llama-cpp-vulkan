"""Vulkan GPU 设备检测与选择.

通过 ggml C API (ctypes) 直接枚举后端设备, 区分独显 (GPU) 和核显 (IGPU),
是独立于 PyTorch/CUDA 的 Vulkan 推理路径.

设备枚举在模块 import 时同步执行(约几百 ms), 属有意设计:
UI 下拉框需要在启动期确定设备列表.
"""

import ctypes
from pathlib import Path

import llama_cpp.llama_cpp as _llama_cpp_lib
from llama_cpp._ggml import (
    GGMLBackendDevType,
    ggml_backend_dev_count,
    ggml_backend_dev_get,
    ggml_backend_load_all_from_path,
    libggml_base,
)

from ..i18n.common_static import AUTO_LABEL, DEVICE_LABEL_TEMPLATE, LOG_PREFIX
from ..i18n.lang import LANG
from ..shared.logger import logger

_LOGS = LANG["logs"]["devices"]

libggml_base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_name.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_description.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_type.restype = ctypes.c_int32

# 设备类型取 wheel 导出的枚举, 消除魔法数与 wheel 升级时的漂移面
_GGML_BACKEND_DEVICE_TYPE_GPU = int(GGMLBackendDevType.GGML_BACKEND_DEVICE_TYPE_GPU)
_GGML_BACKEND_DEVICE_TYPE_IGPU = int(GGMLBackendDevType.GGML_BACKEND_DEVICE_TYPE_IGPU)
_DEV_TYPE_NAMES = {
    _GGML_BACKEND_DEVICE_TYPE_GPU: "GPU",
    _GGML_BACKEND_DEVICE_TYPE_IGPU: "IGPU",
}

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
        logger.warning(LOG_PREFIX + _LOGS["detection_failed"].format(e=e))
        return []


_gpu_devices = _detect_gpu_devices()

if _gpu_devices:
    _summary = ", ".join(f"{d['name']} ({d['desc']}) [{d['type']}]" for d in _gpu_devices)
    logger.info(LOG_PREFIX + _LOGS["detected_devices"].format(count=len(_gpu_devices), summary=_summary))
else:
    logger.warning(LOG_PREFIX + _LOGS["no_devices"])


def _selectable_devices():
    """按 llama.cpp 收集 model->devices 的规则, 返回 main_gpu 实际可选的设备.

    llama.cpp 只把独显 (type==GPU) 按枚举顺序加入设备列表;
    仅当系统没有任何独显时, 才加入第一个核显 (IGPU).
    其余设备无法通过 main_gpu 参数选中, 因此不在下拉框中展示.
    """
    dgpus = [d for d in _gpu_devices if d["type"] == "GPU"]
    if dgpus:
        return dgpus
    return [d for d in _gpu_devices if d["type"] == "IGPU"][:1]


def _device_label(dev):
    return DEVICE_LABEL_TEMPLATE.format(name=dev["name"], desc=dev["desc"], type=dev["type"])


gpu_device_choices = [AUTO_LABEL] + [_device_label(d) for d in _selectable_devices()]


def resolve_device_selection(gpu_device):
    """把下拉框选项翻译为 (main_gpu, split_mode).

    main_gpu 仅在 split_mode=NONE 时生效, 且索引是相对 llama.cpp 的
    model->devices 列表(即 _selectable_devices 的顺序), 不是 ggml 全局设备序号.
    Auto 保持 llama.cpp 默认行为: LAYER 模式, 独显优先, 多独显按层切分.
    """
    if gpu_device == AUTO_LABEL:
        return 0, SPLIT_MODE_LAYER

    for i, dev in enumerate(_selectable_devices()):
        if _device_label(dev) == gpu_device:
            return i, SPLIT_MODE_NONE

    # 防御分支: 选项列表与 _selectable_devices 同源且进程内静态, 跨进程的
    # 过期 label(硬件/驱动变更后的旧工作流)会先被 ComfyUI 对 combo 输入的
    # 前置校验(value_not_in_list)拒绝, 正常执行走不到这里; 未知值按 Auto
    # 语义回退并留警告
    logger.warning(LOG_PREFIX + _LOGS["device_not_selectable"].format(gpu_device=gpu_device))
    return 0, SPLIT_MODE_LAYER


def log_backend_summary(main_gpu, split_mode):
    selectable = _selectable_devices()
    if not selectable:
        logger.warning(LOG_PREFIX + _LOGS["no_backend"])
        return
    if split_mode == SPLIT_MODE_LAYER and len(selectable) > 1:
        names = ", ".join(d["name"] for d in selectable)
        logger.info(LOG_PREFIX + _LOGS["active_gpus_layer_split"].format(names=names))
    else:
        active = selectable[main_gpu] if main_gpu < len(selectable) else selectable[0]
        logger.info(LOG_PREFIX + _LOGS["active_gpu"].format(name=active["name"], desc=active["desc"], type=active["type"]))
