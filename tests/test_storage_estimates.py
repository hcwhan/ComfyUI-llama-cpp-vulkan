"""src/core/storage.py 显存折算函数的单元测试: n_gpu_layers 折算与显存需求估算的边界行为."""

import os
import struct
import tempfile
import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.core.gguf_layers import get_model_meta  # noqa: E402
from src.core.storage import (  # noqa: E402
    _BASE_OVERHEAD,
    _estimate_kv_bytes,
    _estimate_n_gpu_layers,
    _estimate_vram_bytes,
)
from src.i18n.lang import LANG  # noqa: E402

_GB = 1024**3


# 生产路径 (load_model) 解析一次 meta 后传入各估算函数, 测试以同款方式组装参数
def _n_gpu_layers(model_path, mmproj_path, vram_limit, n_ctx):
    return _estimate_n_gpu_layers(model_path, get_model_meta(model_path), mmproj_path, vram_limit, n_ctx)


def _vram_bytes(model_path, mmproj_path, n_gpu_layers, n_ctx):
    return _estimate_vram_bytes(model_path, get_model_meta(model_path), mmproj_path, n_gpu_layers, n_ctx)


def _kv_u32(key, value):
    raw = key.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + struct.pack("<I", 4) + struct.pack("<I", value)


def _gguf_bytes(kv_blobs):
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", len(kv_blobs))
    return header + b"".join(kv_blobs)


def _minimal_gguf_bytes(block_count):
    """构造只含 block_count 元数据的最小 GGUF 文件体(KV 精确计算走不通, 落体积折算回退)."""
    return _gguf_bytes([_kv_u32("llama.block_count", block_count)])


class TestEstimateNGpuLayers(unittest.TestCase):
    def _write_temp(self, data):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def _model_path(self, block_count=32, pad_to_bytes=None):
        data = _minimal_gguf_bytes(block_count)
        if pad_to_bytes is not None:
            data += b"\x00" * max(0, pad_to_bytes - len(data))
        return self._write_temp(data)

    def _write_sparse(self, size):
        """按 size 截断的占位文件: 估算函数只用 os.path.getsize, 无需真实写入 size 字节."""
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.seek(size - 1)
            f.write(b"\x00")
        self.addCleanup(os.remove, path)
        return path

    def _sparse_model(self, block_count, size):
        """带 GGUF 头的按 size 截断的大体积占位模型文件: 层数可控, 体积由 getsize 决定."""
        data = _minimal_gguf_bytes(block_count)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.seek(size - 1)
            f.write(b"\x00")
        self.addCleanup(os.remove, path)
        return path

    def test_minus_one_passthrough_auto(self):
        self.assertEqual(_n_gpu_layers(self._model_path(), None, -1, 8192), (-1, False))

    def test_zero_means_pure_cpu(self):
        self.assertEqual(_n_gpu_layers(self._model_path(), None, 0, 8192), (0, False))

    def test_budget_fits_all_layers_capped_by_estimate(self):
        # 32 层 x 32MB 文件, 预算远大于折算体积, 折算层数应不低于实际层数(超出部分由 llama.cpp 截断)
        path = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        n_layers, _mmproj_on_gpu = _n_gpu_layers(path, None, 8, 8192)
        self.assertGreaterEqual(n_layers, 32)

    def test_mmproj_exceeding_budget_keeps_all_on_cpu(self):
        # 回归 (严格守预算): mmproj 2GB > 预算 1GB, 主模型全留 CPU 且 mmproj 不进显存
        model = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        mmproj = self._write_sparse(2 * _GB)
        self.assertEqual(_n_gpu_layers(model, mmproj, 1, 8192), (0, False))

    def test_mmproj_within_budget_goes_to_gpu(self):
        # mmproj 体积在预算内时正常进显存, 主模型至少 1 层
        model = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        mmproj = self._write_sparse(1 * _GB)
        n_layers, mmproj_on_gpu = _n_gpu_layers(model, mmproj, 8, 8192)
        self.assertGreaterEqual(n_layers, 1)
        self.assertTrue(mmproj_on_gpu)

    def test_budget_below_one_layer_stays_on_cpu(self):
        # 回归 (严格守预算): 预算低于单层折算体积时返回 0 层,
        # 不再强制 1 层突破 vram_limit 上限
        path = self._sparse_model(block_count=2, size=2 * _GB)  # 每层折算约 1.55 GB
        self.assertEqual(_n_gpu_layers(path, None, 1, 8192), (0, False))

    def test_mmproj_fits_but_no_layer_budget_keeps_model_on_cpu(self):
        # mmproj 在预算内照常进显存, 扣除后不足主模型 1 层时主模型留 CPU
        model = self._sparse_model(block_count=2, size=2 * _GB)
        mmproj = self._write_sparse(_GB // 2)  # 折算约 0.575 GB, 预算剩 1.425 GB < 1 层
        self.assertEqual(_n_gpu_layers(model, mmproj, 2, 8192), (0, True))

    def test_kv_fallback_logs_warning_once_per_load(self):
        # 注意力元数据不全导致 KV 降级体积粗估时打 warning; 同一次加载
        # (共享同一份 meta) 内层数折算与腾挪估算多次触发也只打一条
        path = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        meta = get_model_meta(path)
        with self.assertLogs("llama-cpp-vulkan", level="WARNING") as cm:
            _estimate_n_gpu_layers(path, meta, None, 8, 8192)
            _estimate_vram_bytes(path, meta, None, -1, 8192)
        fallback_lines = [line for line in cm.output if LANG["logs"]["storage"]["kv_meta_fallback"] in line]
        self.assertEqual(len(fallback_lines), 1)

    def test_kv_fallback_warns_on_auto_via_vram_bytes(self):
        # vram_limit=-1 (auto) 不经层数折算, 降级警告经腾挪估算路径同样发出
        path = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        with self.assertLogs("llama-cpp-vulkan", level="WARNING") as cm:
            _vram_bytes(path, None, -1, 8192)
        self.assertTrue(any(LANG["logs"]["storage"]["kv_meta_fallback"] in line for line in cm.output))

    def test_no_kv_fallback_warning_when_metadata_complete(self):
        data = _gguf_bytes(
            [
                _kv_u32("llama.block_count", 32),
                _kv_u32("llama.embedding_length", 4096),
                _kv_u32("llama.attention.head_count", 32),
                _kv_u32("llama.attention.head_count_kv", 8),
            ]
        )
        data += b"\x00" * (32 * 1024 * 1024 - len(data))
        path = self._write_temp(data)
        with self.assertNoLogs("llama-cpp-vulkan", level="WARNING"):
            _n_gpu_layers(path, None, 8, 8192)
            _vram_bytes(path, None, -1, 8192)

    def test_kv_fallback_silent_on_cpu_shortcut(self):
        # vram_limit=0 (纯 CPU) 不做任何估算, 元数据不全也不打降级 warning
        path = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        with self.assertNoLogs("llama-cpp-vulkan", level="WARNING"):
            _n_gpu_layers(path, None, 0, 8192)


class TestEstimateKvBytes(unittest.TestCase):
    def test_exact_kv_from_metadata(self):
        # 8192 ctx x 32 层 x 8 kv头 x (128+128) 维 x 2 字节(f16) = 1 GB
        meta = {"head_count_kv": 8, "head_count": 32, "embedding_length": 4096}
        self.assertEqual(_estimate_kv_bytes(meta, 32, 8192), _GB)

    def test_array_kv_heads_averaged(self):
        # hybrid 模型逐层 head_count_kv(线性注意力层为 0), 取均值折算
        meta = {"head_count_kv": [0, 8, 0, 8], "head_count": 32, "embedding_length": 4096}
        self.assertEqual(_estimate_kv_bytes(meta, 32, 8192), _GB // 2)

    def test_explicit_key_value_length_override(self):
        # key_length/value_length 存在时优先于 embedding/head_count 推导
        meta = {"head_count_kv": 8, "key_length": 64, "value_length": 32}
        expected = 8192 * 32 * 8 * (64 + 32) * 2
        self.assertEqual(_estimate_kv_bytes(meta, 32, 8192), expected)

    def test_missing_metadata_returns_none(self):
        self.assertIsNone(_estimate_kv_bytes({}, 32, 8192))
        self.assertIsNone(_estimate_kv_bytes({"head_count_kv": 8}, 32, 8192))

    def test_zero_kv_heads_returns_zero(self):
        # 纯 recurrent 架构 (纯 Mamba 类): head_count_kv 明确为 0 表示无 KV cache,
        # 即使缺少其余注意力字段也返回 0, 不落含 KV 项的体积折算回退
        self.assertEqual(_estimate_kv_bytes({"head_count_kv": 0}, 32, 8192), 0)

    def test_all_zero_array_kv_heads_returns_zero(self):
        # 逐层 head_count_kv 全 0 (数组均值 0.0) 与标量 0 同义
        self.assertEqual(_estimate_kv_bytes({"head_count_kv": [0, 0, 0, 0]}, 32, 8192), 0)

    def test_swa_window_reduces_kv(self):
        # SWA 模型 (如 Gemma3) 的滑窗层只分配约 (窗口 + n_ubatch) 的 KV,
        # 按半数层 SWA 的保守占比折算: 有效 token = (8192 + 512 + 512) / 2 = 4608
        base = {"head_count_kv": 8, "head_count": 32, "embedding_length": 4096}
        meta = dict(base, sliding_window=512)
        expected = int((8192 + 512 + 512) / 2 * 32 * 8 * (128 + 128) * 2)
        self.assertEqual(_estimate_kv_bytes(meta, 32, 8192), expected)
        self.assertLess(_estimate_kv_bytes(meta, 32, 8192), _estimate_kv_bytes(base, 32, 8192))

    def test_swa_window_covering_ctx_no_reduction(self):
        # 窗口 (含批处理余量) 不小于 n_ctx 时滑窗不生效, 按全量 n_ctx 计
        base = {"head_count_kv": 8, "head_count": 32, "embedding_length": 4096}
        meta = dict(base, sliding_window=4096)
        self.assertEqual(_estimate_kv_bytes(meta, 32, 4096), _estimate_kv_bytes(base, 32, 4096))


class TestEstimateVramBytes(unittest.TestCase):
    def _write_temp(self, data):
        fd, path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        self.addCleanup(os.remove, path)
        return path

    def setUp(self):
        data = _minimal_gguf_bytes(32) + b"\x00" * (1024 * 1024)
        self.model = self._write_temp(data[: 1024 * 1024])
        self.mmproj = self._write_temp(b"\x00" * (512 * 1024))

    def test_zero_layers_counts_only_mmproj(self):
        # 主模型 0 层不进显存, 估算只含 mmproj 体积
        size = _vram_bytes(self.model, self.mmproj, 0, 8192)
        self.assertLess(size, os.path.getsize(self.model))
        self.assertGreater(size, 0)

    def test_auto_layers_counts_full_model(self):
        size = _vram_bytes(self.model, None, -1, 8192)
        self.assertGreaterEqual(size, os.path.getsize(self.model))

    def test_precise_kv_used_when_metadata_present(self):
        # 注意力元数据齐全时, 估算 = 体积 x (1+固定开销) + 精确 KV 字节数
        data = _gguf_bytes(
            [
                _kv_u32("llama.block_count", 2),
                _kv_u32("llama.embedding_length", 64),
                _kv_u32("llama.attention.head_count", 4),
                _kv_u32("llama.attention.head_count_kv", 2),
            ]
        )
        data += b"\x00" * (1024 * 1024 - len(data))
        model = self._write_temp(data)
        kv = 8192 * 2 * 2 * (16 + 16) * 2
        expected = int(os.path.getsize(model) * (1.0 + _BASE_OVERHEAD) + kv)
        self.assertEqual(_vram_bytes(model, None, -1, 8192), expected)

    def test_zero_kv_heads_skips_fallback(self):
        # 纯 recurrent GGUF (head_count_kv=0): 按 KV=0 精确估算, 不落体积
        # 折算回退, 也不打降级 warning (元数据其实是全的)
        data = _gguf_bytes(
            [
                _kv_u32("mamba.block_count", 2),
                _kv_u32("mamba.attention.head_count_kv", 0),
            ]
        )
        data += b"\x00" * (1024 * 1024 - len(data))
        model = self._write_temp(data)
        expected = int(os.path.getsize(model) * (1.0 + _BASE_OVERHEAD))
        with self.assertNoLogs("llama-cpp-vulkan", level="WARNING"):
            self.assertEqual(_vram_bytes(model, None, -1, 8192), expected)


if __name__ == "__main__":
    unittest.main()
