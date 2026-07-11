"""GGUF 文件头解析, 只读取元数据中的 block_count(模型层数), 供显存折算使用."""

import struct

from ..shared.logger import logger

# GGUF 标量 value type -> struct 格式 (string=8 和 array=9 单独处理)
_SCALAR_FORMATS = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}


def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    ln = read_u64(f)
    # BPE 模型的 tokenizer 元数据可能含非法 UTF-8 字节序列，严格解码会中断整个解析
    return f.read(ln).decode("utf-8", errors="replace")


def _read_scalar(f, vtype):
    fmt = _SCALAR_FORMATS.get(vtype)
    if fmt is not None:
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == 8:   # string
        return read_string(f)
    raise ValueError(f"Unknown value type {vtype}")


def read_value(f):
    vtype = read_u32(f)
    if vtype == 9:   # array (元素只能是标量,GGUF 不支持嵌套数组)
        atype = read_u32(f)
        count = read_u64(f)
        return [_read_scalar(f, atype) for _ in range(count)]
    return _read_scalar(f, vtype)

def _parse_block_count(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("This is not a GGUF file!")

        _version = read_u32(f)
        _tensor_count = read_u64(f)
        kv_count = read_u64(f)

        # block_count 通常排在 tokenizer 数组之前，命中即返回，
        # 避免解析几十万条 token 元数据（慢且占内存）
        for _ in range(kv_count):
            key = read_string(f)
            value = read_value(f)
            if key.lower().endswith(".block_count"):
                return value

    return None


def get_layer_count(path):
    """读取 GGUF 模型层数,失败返回 None(调用方回退默认值)。"""
    try:
        count = _parse_block_count(path)
        if count is None:
            logger.warning("[llama-cpp-vulkan] block_count not found in GGUF metadata")
        return count
    except Exception as e:
        logger.warning(f"[llama-cpp-vulkan] GGUF parse failed: {e}")
        return None
