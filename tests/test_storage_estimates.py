"""app/core/storage.py 显存折算函数的单元测试: n_gpu_layers 折算与显存需求估算的边界行为."""

import os
import struct
import tempfile
import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.core.storage import _estimate_n_gpu_layers, _estimate_vram_bytes  # noqa: E402

_GB = 1024 ** 3


def _minimal_gguf_bytes(block_count):
    """构造只含 block_count 元数据的最小 GGUF 文件体."""
    key = b"llama.block_count"
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1)
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", block_count)
    return header + kv


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

    def test_minus_one_passthrough_auto(self):
        self.assertEqual(_estimate_n_gpu_layers(self._model_path(), None, -1, 8192), -1)

    def test_zero_means_pure_cpu(self):
        self.assertEqual(_estimate_n_gpu_layers(self._model_path(), None, 0, 8192), 0)

    def test_budget_fits_all_layers_capped_by_estimate(self):
        # 32 层 x 32MB 文件, 预算远大于折算体积, 折算层数应超过实际层数(由 llama.cpp 截断)
        path = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        self.assertGreaterEqual(_estimate_n_gpu_layers(path, None, 8, 8192), 32)

    def test_mmproj_exceeding_budget_returns_zero(self):
        # mmproj 2GB > 预算 1GB, 主模型应全留 CPU 而不是强塞 1 层
        model = self._model_path(block_count=32, pad_to_bytes=32 * 1024 * 1024)
        mmproj = self._write_temp(b"\x00" * (2 * _GB // 1024) * 1024)
        self.assertEqual(_estimate_n_gpu_layers(model, mmproj, 1, 8192), 0)

    def test_tiny_positive_budget_keeps_at_least_one_layer(self):
        path = self._model_path(block_count=32, pad_to_bytes=64 * 1024 * 1024)
        self.assertGreaterEqual(_estimate_n_gpu_layers(path, None, 1, 8192), 1)


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
        size = _estimate_vram_bytes(self.model, self.mmproj, 0, 8192)
        self.assertLess(size, os.path.getsize(self.model))
        self.assertGreater(size, 0)

    def test_auto_layers_counts_full_model(self):
        size = _estimate_vram_bytes(self.model, None, -1, 8192)
        self.assertGreaterEqual(size, os.path.getsize(self.model))


if __name__ == "__main__":
    unittest.main()
