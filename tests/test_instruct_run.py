"""src/core/instruct.py 执行骨架的单元测试: InterruptWatcher 中断链路与 _run 收尾分支.

覆盖: watcher 命中后置位并持续 abort, 无中断时干净收线; _run 的 finally
收尾 (force_offload 卸载, hybrid 三件套重置, 中断丢弃截断结果抛
InterruptProcessingException, runner 异常时收尾仍执行).
"""

import time
import types
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import instruct  # noqa: E402
from src.core.instruct import InterruptWatcher, llama_cpp_instruct_base  # noqa: E402
from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402


class _AbortRecorder:
    def __init__(self):
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1


class TestInterruptWatcher(unittest.TestCase):
    def test_interrupt_sets_flag_and_aborts(self):
        llm = _AbortRecorder()
        with (
            mock.patch.object(instruct.mm, "processing_interrupted", lambda: True),
            InterruptWatcher(llm, poll_interval=0.01) as watcher,
        ):
            deadline = time.time() + 2.0
            while not watcher.interrupted and time.time() < deadline:
                time.sleep(0.005)
        self.assertTrue(watcher.interrupted)
        self.assertGreaterEqual(llm.abort_calls, 1)
        self.assertFalse(watcher._thread.is_alive())

    def test_hit_keeps_re_aborting_against_clear_race(self):
        # 命中后持续重复 set: 对抗 create_completion 每次请求开始 clear abort_event
        llm = _AbortRecorder()
        with (
            mock.patch.object(instruct.mm, "processing_interrupted", lambda: True),
            InterruptWatcher(llm, poll_interval=0.01) as watcher,
        ):
            deadline = time.time() + 2.0
            while llm.abort_calls < 3 and time.time() < deadline:
                time.sleep(0.005)
        self.assertGreaterEqual(llm.abort_calls, 3)
        self.assertTrue(watcher.interrupted)

    def test_no_interrupt_exits_cleanly(self):
        llm = _AbortRecorder()
        with InterruptWatcher(llm, poll_interval=0.01) as watcher:
            time.sleep(0.05)
        self.assertFalse(watcher.interrupted)
        self.assertEqual(llm.abort_calls, 0)
        self.assertFalse(watcher._thread.is_alive())


class _FakeRunLlm:
    """_run 收尾分支所需的最小 llm 替身 (hybrid 判定与重置三件套)."""

    def __init__(self, hybrid=False):
        self.n_tokens = 7
        self._model = types.SimpleNamespace(
            is_hybrid=lambda: hybrid,
            is_recurrent=lambda: False,
        )
        self._ctx = mock.Mock()
        self._hybrid_cache_mgr = mock.Mock()

    def abort(self):
        pass


class TestRunFinalization(unittest.TestCase):
    def setUp(self):
        self.node = llama_cpp_instruct_base()
        self.config = {"model": "m.gguf"}
        self._orig_state = (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config)
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config) = self._orig_state

    def _install(self, llm):
        # current_config 与 llama_model 相同, _prepare_messages 不触发真实加载
        LLAMA_CPP_STORAGE.llm = llm
        LLAMA_CPP_STORAGE.chat_handler = None
        LLAMA_CPP_STORAGE.current_config = self.config

    def _run(self, runner, force_offload=False):
        return self.node._run(
            llama_model=self.config,
            preset_prompt="空白 - 空",
            custom_prompt="一只猫",
            system_prompt="",
            seed=0,
            force_offload=force_offload,
            strip_thinking=True,
            parameters=None,
            runner=runner,
        )

    def test_success_returns_runner_output(self):
        self._install(_FakeRunLlm())
        out = self._run(lambda *args: "ok")
        self.assertEqual(out, ("ok",))

    def test_force_offload_cleans_after_success(self):
        self._install(_FakeRunLlm())
        self._run(lambda *args: "ok", force_offload=True)
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)
        self.assertIsNone(LLAMA_CPP_STORAGE.current_config)

    def test_hybrid_arch_reset_after_success(self):
        llm = _FakeRunLlm(hybrid=True)
        self._install(llm)
        self._run(lambda *args: "ok")
        self.assertEqual(llm.n_tokens, 0)
        llm._ctx.memory_clear.assert_called_once_with(True)
        llm._hybrid_cache_mgr.clear.assert_called_once()

    def test_non_hybrid_keeps_kv_cache(self):
        llm = _FakeRunLlm(hybrid=False)
        self._install(llm)
        self._run(lambda *args: "ok")
        self.assertEqual(llm.n_tokens, 7)
        llm._ctx.memory_clear.assert_not_called()

    def test_interrupted_discards_output_and_raises(self):
        # abort_event 使生成提前返回截断结果, 须丢弃并走标准中断流程
        self._install(_FakeRunLlm())

        def runner(messages, user_content, seed, params, extract_text, watcher):
            watcher.interrupted = True
            return "truncated garbage"

        with self.assertRaises(instruct.mm.InterruptProcessingException):
            self._run(runner)

    def test_runner_exception_still_finalizes(self):
        # finally 收尾: 异常路径同样执行 force_offload 卸载
        self._install(_FakeRunLlm())
        with self.assertRaises(RuntimeError):
            self._run(mock.Mock(side_effect=RuntimeError("boom")), force_offload=True)
        self.assertIsNone(LLAMA_CPP_STORAGE.llm)


if __name__ == "__main__":
    unittest.main()
