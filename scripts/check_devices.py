"""独立诊断脚本, 列出 GGML 后端检测到的所有设备(CPU/GPU/IGPU/ACCEL)."""

import ctypes
from pathlib import Path

import llama_cpp.llama_cpp as _llama_cpp_lib
from llama_cpp._ggml import (
    GGMLBackendDevType,
    libggml_base,
    ggml_backend_dev_count,
    ggml_backend_dev_get,
    ggml_backend_load_all_from_path,
    ggml_backend_reg_count,
)

# 由 wheel 导出的枚举生成 {0: "CPU", 1: "GPU", ...}, 随 wheel 升级自动同步
DEV_TYPES = {int(t): t.name.removeprefix("GGML_BACKEND_DEVICE_TYPE_") for t in GGMLBackendDevType}

lib_dir = Path(_llama_cpp_lib.__file__).resolve().parent / "lib"
print(f"Loading backends from: {lib_dir}\n")
ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))

print(f"Backend registrations: {ggml_backend_reg_count()}")

libggml_base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_name.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_description.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_type.restype = ctypes.c_int32

count = ggml_backend_dev_count()
print(f"Total devices: {count}\n")
for i in range(count):
    dev = ggml_backend_dev_get(i)
    name = libggml_base.ggml_backend_dev_name(dev).decode("utf-8", errors="replace")
    desc = libggml_base.ggml_backend_dev_description(dev).decode("utf-8", errors="replace").strip()
    dev_type = libggml_base.ggml_backend_dev_type(dev)
    type_str = DEV_TYPES.get(dev_type, f"UNKNOWN({dev_type})")
    print(f"[{i}] {type_str}: {name} ({desc})")
