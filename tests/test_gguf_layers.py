"""src/core/gguf_layers.py 的单元测试: 用手工构造的最小 GGUF 文件验证解析."""

import os
import struct
import tempfile
import unittest

from src.core.gguf_layers import get_model_meta


def _block_count(path):
    # 生产路径 (storage._estimate_per_layer_bytes) 取层数的同款写法:
    # 解析失败 get_model_meta 返回空 dict, block_count 缺失得 None
    return get_model_meta(path).get("block_count")


# GGUF value type 编号(与被测模块一致)
_T_UINT32 = 4
_T_STRING = 8
_T_ARRAY = 9


def _string(s):
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_uint32(key, value):
    return _string(key) + struct.pack("<I", _T_UINT32) + struct.pack("<I", value)


def _kv_string(key, value):
    return _string(key) + struct.pack("<I", _T_STRING) + _string(value)


def _kv_string_array(key, values):
    body = struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_STRING) + struct.pack("<Q", len(values))
    for v in values:
        body += _string(v)
    return _string(key) + body


def _gguf_bytes(kv_blobs):
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv_blobs))
    return header + b"".join(kv_blobs)


class TestBlockCountParsing(unittest.TestCase):
    def _write_temp(self, data):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def test_block_count_found(self):
        path = self._write_temp(
            _gguf_bytes(
                [
                    _kv_string("general.architecture", "llama"),
                    _kv_uint32("llama.block_count", 32),
                ]
            )
        )
        self.assertEqual(_block_count(path), 32)

    def test_block_count_after_array_kv(self):
        # 命中前需要正确跳过数组类型的 KV
        path = self._write_temp(
            _gguf_bytes(
                [
                    _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
                    _kv_uint32("qwen2.block_count", 48),
                ]
            )
        )
        self.assertEqual(_block_count(path), 48)

    def test_block_count_missing_returns_none(self):
        path = self._write_temp(
            _gguf_bytes(
                [
                    _kv_string("general.architecture", "llama"),
                ]
            )
        )
        self.assertIsNone(_block_count(path))

    def test_non_gguf_file_returns_none(self):
        path = self._write_temp(b"NOT A GGUF FILE")
        self.assertIsNone(_block_count(path))

    def test_truncated_file_returns_none(self):
        path = self._write_temp(b"GGUF" + struct.pack("<I", 3))
        self.assertIsNone(_block_count(path))

    def test_implausible_array_count_falls_back(self):
        # 回归: KV 区错位使数组 count 读成天文数字时, 须立即报错走回退
        # (返回空 dict), 而不是以几字节步长扫过整个文件打转数分钟
        huge_array = (
            _string("tokenizer.ggml.tokens") + struct.pack("<I", _T_ARRAY) + struct.pack("<I", _T_UINT32) + struct.pack("<Q", 10**12)
        )
        path = self._write_temp(_gguf_bytes([huge_array, _kv_uint32("llama.block_count", 32)]))
        self.assertIsNone(_block_count(path))

    def test_gguf_v1_rejected(self):
        # v1 的计数字段是 32 位, 按 v2+ 布局读会错乱, 应直接按不支持返回 None
        data = b"GGUF" + struct.pack("<I", 1) + struct.pack("<I", 0) + struct.pack("<I", 1)
        path = self._write_temp(data + _kv_uint32("llama.block_count", 32))
        self.assertIsNone(_block_count(path))


if __name__ == "__main__":
    unittest.main()
