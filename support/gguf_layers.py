import struct

def read_u32(f):
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f):
    ln = read_u64(f)
    # BPE 模型的 tokenizer 元数据可能含非法 UTF-8 字节序列，严格解码会中断整个解析
    return f.read(ln).decode("utf-8", errors="replace")


def read_value(f):
    vtype = read_u32(f)

    # GGUF value types
    if vtype == 0:   # uint8
        return struct.unpack("<B", f.read(1))[0]
    if vtype == 1:   # int8
        return struct.unpack("<b", f.read(1))[0]
    if vtype == 2:   # uint16
        return struct.unpack("<H", f.read(2))[0]
    if vtype == 3:   # int16
        return struct.unpack("<h", f.read(2))[0]
    if vtype == 4:   # uint32
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 5:   # int32
        return struct.unpack("<i", f.read(4))[0]
    if vtype == 6:   # float32
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:   # bool
        return struct.unpack("<?", f.read(1))[0]
    if vtype == 8:   # string
        return read_string(f)
    if vtype == 9:   # array
        atype = read_u32(f)
        count = read_u64(f)
        return [read_value_of_type(f, atype) for _ in range(count)]
    if vtype == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    if vtype == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    if vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"Unknown value type {vtype}")


def read_value_of_type(f, atype):
    # same mapping as above but without extra type code
    if atype == 0:
        return struct.unpack("<B", f.read(1))[0]
    if atype == 1:
        return struct.unpack("<b", f.read(1))[0]
    if atype == 2:
        return struct.unpack("<H", f.read(2))[0]
    if atype == 3:
        return struct.unpack("<h", f.read(2))[0]
    if atype == 4:
        return struct.unpack("<I", f.read(4))[0]
    if atype == 5:
        return struct.unpack("<i", f.read(4))[0]
    if atype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if atype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if atype == 8:
        return read_string(f)
    if atype == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if atype == 11:
        return struct.unpack("<q", f.read(8))[0]
    if atype == 12:
        return struct.unpack("<d", f.read(8))[0]

    raise ValueError(f"Unknown array item type {atype}")

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
    try:
        count = _parse_block_count(path)
        if count is not None:
            return count
        print("[gguf_layers] block_count not found in metadata, trying GGUFReader fallback...")
    except Exception as e:
        print(f"[gguf_layers] manual GGUF parse failed ({e}), trying GGUFReader fallback...")

    try:
        from gguf import GGUFReader
        reader = GGUFReader(path)

        # get_field 返回 ReaderField 而非数值，直接 int() 会取到 offset 等错误值；
        # 统一通过 parts[data[0]] 读取实际数据
        layer_field = reader.get_field("llama.block_count")
        if layer_field is None:
            for field in reader.fields.values():
                if field.name.endswith(".block_count"):
                    layer_field = field
                    break

        if layer_field is not None:
            return int(layer_field.parts[layer_field.data[0]][0])
    except Exception as e:
        print(f"Failed to get block_count: {e}")

    return None