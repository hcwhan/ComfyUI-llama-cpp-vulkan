"""src/core/storage.py load_model/clean 状态机的单元测试 (mock Llama, 不加载真实模型).

覆盖: 校验失败不动已加载模型, 加载失败清理半初始化状态, 重试一轮路径,
加载前与重试前的中断响应, 成功后才写 current_config,
free_memory 腾挪的触发条件, clean 的幂等性.
"""

import os
import struct
import tempfile
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import storage  # noqa: E402
from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.i18n.common_static import AUTO_LABEL  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402


def _minimal_gguf_bytes(block_count=4):
    key = b"llama.block_count"
    kv = struct.pack("<Q", len(key)) + key + struct.pack("<I", 4) + struct.pack("<I", block_count)
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 1) + kv


class _FakeLlama:
    """构造可控失败次数的 Llama 替身, close 只做记录."""

    fail_remaining = 0

    def __init__(self, model_path, **kwargs):
        if _FakeLlama.fail_remaining > 0:
            _FakeLlama.fail_remaining -= 1
            raise RuntimeError("simulated load failure")
        self.model_path = model_path
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


class TestLoadModelStateMachine(unittest.TestCase):
    def setUp(self):
        fd, self.model_path = tempfile.mkstemp(suffix=".gguf")
        with os.fdopen(fd, "wb") as f:
            f.write(_minimal_gguf_bytes())
        self.addCleanup(os.remove, self.model_path)

        _FakeLlama.fail_remaining = 0
        patches = [
            mock.patch.object(storage, "Llama", _FakeLlama),
            mock.patch.object(
                storage,
                "get_llm_full_path",
                lambda name: self.model_path if name == "model.gguf" else None,
            ),
            # 重试路径的 WDDM 等待与测试无关, 打桩省掉 1 秒
            mock.patch.object(storage.time, "sleep", lambda s: None),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        self.free_memory = mock.Mock(return_value=[])
        p = mock.patch.object(storage.mm, "free_memory", self.free_memory)
        p.start()
        self.addCleanup(p.stop)

        # 隔离全局单例状态, 用例间与用例后互不污染
        self._orig_state = (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config)
        LLAMA_CPP_STORAGE.llm = None
        LLAMA_CPP_STORAGE.chat_handler = None
        LLAMA_CPP_STORAGE.current_config = None
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config) = self._orig_state

    @staticmethod
    def _config(vram_limit=-1):
        return {
            "gpu_device": AUTO_LABEL,
            "model": "model.gguf",
            "mmproj": "None",
            "chat_handler": "None",
            "thinking": False,
            "n_ctx": 2048,
            "vram_limit": vram_limit,
            "image_min_tokens": 0,
            "image_max_tokens": 0,
        }

    def test_success_records_config_copy(self):
        config = self._config()
        LLAMA_CPP_STORAGE.load_model(config)
        self.assertIsInstance(LLAMA_CPP_STORAGE.llm, _FakeLlama)
        self.assertEqual(LLAMA_CPP_STORAGE.current_config, config)
        # 记录副本: 调用方后续改 dict 不得影响 "无需重载" 判定
        self.assertIsNot(LLAMA_CPP_STORAGE.current_config, config)

    def test_invalid_config_keeps_current_model(self):
        # 校验先于卸载: 无效新配置不得影响已加载的模型
        LLAMA_CPP_STORAGE.load_model(self._config())
        loaded = LLAMA_CPP_STORAGE.llm
        bad = dict(self._config(), model="missing.gguf")
        with self.assertRaises(FileNotFoundError):
            LLAMA_CPP_STORAGE.load_model(bad)
        self.assertIs(LLAMA_CPP_STORAGE.llm, loaded)
        self.assertFalse(loaded.closed)
        self.assertEqual(LLAMA_CPP_STORAGE.current_config["model"], "model.gguf")

    def test_load_failure_cleans_state_after_retry(self):
        # 首次与重试均失败: 异常传播且不残留半初始化状态与新配置
        _FakeLlama.fail_remaining = 2
        with self.assertRaises(RuntimeError):
            LLAMA_CPP_STORAGE.load_model(self._config())
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.chat_handler)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)
        # 首次加载前腾挪一次 + 重试前再腾挪一次
        self.assertEqual(self.free_memory.call_count, 2)

    def test_transient_failure_retried_once(self):
        _FakeLlama.fail_remaining = 1
        config = self._config()
        LLAMA_CPP_STORAGE.load_model(config)
        self.assertIsInstance(LLAMA_CPP_STORAGE.llm, _FakeLlama)
        self.assertEqual(LLAMA_CPP_STORAGE.current_config, config)

    def test_cpu_only_failure_not_retried(self):
        # 纯 CPU 加载失败与显存无关, 不做重试 (fail_remaining 只被消耗一次)
        _FakeLlama.fail_remaining = 1
        with self.assertRaises(RuntimeError):
            LLAMA_CPP_STORAGE.load_model(self._config(vram_limit=0))
        self.assertEqual(_FakeLlama.fail_remaining, 0)
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)

    def test_cpu_only_skips_free_memory(self):
        LLAMA_CPP_STORAGE.load_model(self._config(vram_limit=0))
        self.free_memory.assert_not_called()
        self.assertEqual(LLAMA_CPP_STORAGE.llm.kwargs["n_gpu_layers"], 0)

    def test_interrupt_before_load_raises_with_clean_state(self):
        # 排队期间点 Cancel: 加载真正开始前响应中断,
        # 此时旧模型已卸载而新模型未构造, 状态干净
        with (
            mock.patch.object(storage.mm, "processing_interrupted", lambda: True),
            self.assertRaises(storage.mm.InterruptProcessingException),
        ):
            LLAMA_CPP_STORAGE.load_model(self._config())
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.chat_handler)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)

    def test_interrupt_before_retry_raises_with_clean_state(self):
        # 首次加载期间点 Cancel: 重试前响应中断, 不再白费一次全量加载;
        # 加载前检查(第一次调用)须放行, 故 side_effect 先 False 后 True
        _FakeLlama.fail_remaining = 1
        with (
            mock.patch.object(storage.mm, "processing_interrupted", mock.Mock(side_effect=[False, True])),
            self.assertRaises(storage.mm.InterruptProcessingException),
        ):
            LLAMA_CPP_STORAGE.load_model(self._config())
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.chat_handler)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)
        # 重试未发生: 首次加载前腾挪一次后即中止, 无重试前的第二次腾挪
        self.assertEqual(self.free_memory.call_count, 1)

    def test_cpu_only_logs_cpu_message(self):
        # 纯 CPU 推理 (0 层且 mmproj 不上卡) 打 cpu_only, 不打 "启用的 GPU"
        with (
            mock.patch.object(storage, "log_backend_summary") as summary,
            self.assertLogs("llama-cpp-vulkan", level="INFO") as logs,
        ):
            LLAMA_CPP_STORAGE.load_model(self._config(vram_limit=0))
        summary.assert_not_called()
        self.assertTrue(any(LANG["logs"]["storage"]["cpu_only"] in m for m in logs.output))

    def test_mmproj_only_on_gpu_logs_dedicated_message(self):
        # 回归: (0 层, mmproj 进显存) 组合 (vram_no_room_for_layer 分支) 不得打
        # "启用的 GPU" (主模型不在 GPU 上, 且 mmproj 落点由 mtmd 自选), 打专门日志
        with (
            mock.patch.object(storage, "_estimate_n_gpu_layers", lambda *a: (0, True)),
            mock.patch.object(storage, "log_backend_summary") as summary,
            self.assertLogs("llama-cpp-vulkan", level="INFO") as logs,
        ):
            LLAMA_CPP_STORAGE.load_model(self._config(vram_limit=1))
        summary.assert_not_called()
        self.assertTrue(any(LANG["logs"]["storage"]["mmproj_only_gpu"] in m for m in logs.output))

    def test_gpu_layers_log_backend_summary(self):
        # 有层上卡时走 log_backend_summary (单卡/多卡文案由 devices.py 内部分支)
        with mock.patch.object(storage, "log_backend_summary") as summary:
            LLAMA_CPP_STORAGE.load_model(self._config())
        summary.assert_called_once()

    def test_clean_closes_and_is_idempotent(self):
        LLAMA_CPP_STORAGE.load_model(self._config())
        loaded = LLAMA_CPP_STORAGE.llm
        LLAMA_CPP_STORAGE.clean()
        self.assertTrue(loaded.closed)
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.chat_handler)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)
        LLAMA_CPP_STORAGE.clean()

    def test_clean_when_empty_skips_gc(self):
        # 空载 clean (Unload 节点空跑兜底等) 只归零配置即返回,
        # 不触发全量 gc.collect (大进程中单次可达几十毫秒)
        LLAMA_CPP_STORAGE.current_config = {"stale": True}
        with mock.patch.object(storage.gc, "collect") as collect:
            LLAMA_CPP_STORAGE.clean()
        collect.assert_not_called()
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)

    def test_clean_after_load_runs_gc(self):
        # 真实卸载路径仍执行 gc.collect, 及时回收 ctypes 资源引用
        LLAMA_CPP_STORAGE.load_model(self._config())
        with mock.patch.object(storage.gc, "collect") as collect:
            LLAMA_CPP_STORAGE.clean()
        collect.assert_called_once()

    def test_unload_hook_cleans_storage(self):
        # monkey-patch 后的 mm.unload_all_models 经 sys.modules 动态取当前
        # 生效模块的类 (热重载加固), 调用即清理单例并级联原函数
        LLAMA_CPP_STORAGE.load_model(self._config())
        loaded = LLAMA_CPP_STORAGE.llm
        storage.mm.unload_all_models()
        self.assertTrue(loaded.closed)
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)


if __name__ == "__main__":
    unittest.main()
