"""GGUF 文件头解析, 读取显存折算所需的元数据(层数与注意力 KV 参数)."""

import struct

from ..shared.logger import logger

# GGUF 标量 value type -> struct 格式 (string=8 和 array=9 单独处理)
_SCALAR_FORMATS = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<?",  # bool
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
    if vtype == 8:  # string
        return read_string(f)
    raise ValueError(f"Unknown value type {vtype}")


def read_value(f):
    vtype = read_u32(f)
    if vtype == 9:  # array (元素只能是标量,GGUF 不支持嵌套数组)
        atype = read_u32(f)
        count = read_u64(f)
        return [_read_scalar(f, atype) for _ in range(count)]
    return _read_scalar(f, vtype)


# 显存折算所需的元数据字段 -> GGUF key 后缀(实际 key 带架构名前缀, 按后缀匹配);
# key_length/value_length 仅部分模型存在(head_dim 与 embedding/head_count 不一致时)
_META_SUFFIXES = {
    "block_count": ".block_count",
    "embedding_length": ".embedding_length",
    "head_count": ".attention.head_count",
    "head_count_kv": ".attention.head_count_kv",
    "key_length": ".attention.key_length",
    "value_length": ".attention.value_length",
}


def _parse_metadata(path):
    """顺序扫描 GGUF KV 区, 返回 {字段名: 值}, 字段集齐即提前返回.

    架构键通常排在 tokenizer 数组之前; 关键的 block_count 到手后一旦扫到
    tokenizer 区即停止, 不为可选字段解析几十万条 token 元数据(慢且占内存).
    """
    found = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("This is not a GGUF file!")

        version = read_u32(f)
        if version < 2:
            # v1 的 tensor/kv 计数与字符串长度是 32 位, 按下面的 64 位布局读会
            # 解析错乱; v1 在生态中已基本绝迹, 直接按不支持报错
            raise ValueError(f"GGUF v{version} is too old (v2+ required)")
        _tensor_count = read_u64(f)
        kv_count = read_u64(f)

        for _ in range(kv_count):
            key = read_string(f).lower()
            if key.startswith("tokenizer.") and "block_count" in found:
                break
            value = read_value(f)
            for name, suffix in _META_SUFFIXES.items():
                if name not in found and key.endswith(suffix):
                    found[name] = value
                    break
            if len(found) == len(_META_SUFFIXES):
                break

    return found


def get_model_meta(path):
    """读取显存折算所需的 GGUF 元数据 dict, 解析失败返回空 dict(调用方走回退路径)."""
    try:
        meta = _parse_metadata(path)
        if "block_count" not in meta:
            logger.warning("[llama-cpp-vulkan] block_count not found in GGUF metadata")
        return meta
    except Exception as e:
        logger.warning(f"[llama-cpp-vulkan] GGUF parse failed: {e}")
        return {}
