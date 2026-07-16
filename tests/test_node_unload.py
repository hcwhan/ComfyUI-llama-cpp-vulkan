"""src/nodes/model/node_unload.py 的节点级单元测试: any 透传, clean 调用与日志分支."""

import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402
from src.nodes.model.node_unload import llama_cpp_unload_model  # noqa: E402


class TestUnloadNode(unittest.TestCase):
    def setUp(self):
        self._orig_llm = LLAMA_CPP_STORAGE.llm
        self.addCleanup(setattr, LLAMA_CPP_STORAGE, "llm", self._orig_llm)
        p = mock.patch.object(LLAMA_CPP_STORAGE, "clean")
        self.clean = p.start()
        self.addCleanup(p.stop)

    def test_passthrough_returns_same_object(self):
        # any 透传: 输入对象原样返回 (身份而非相等)
        LLAMA_CPP_STORAGE.llm = None
        payload = object()
        result = llama_cpp_unload_model().process(payload)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], payload)

    def test_loaded_model_logs_and_cleans(self):
        LLAMA_CPP_STORAGE.llm = object()
        with self.assertLogs("llama-cpp-vulkan", level="INFO") as logs:
            llama_cpp_unload_model().process("x")
        self.assertTrue(any(LANG["logs"]["unload"]["unloading"] in m for m in logs.output))
        self.clean.assert_called_once()

    def test_empty_state_skips_log_but_still_cleans(self):
        # 空载不打 "Unloading" 日志 (避免误导排查), clean 仍无条件执行兜底
        LLAMA_CPP_STORAGE.llm = None
        with self.assertNoLogs("llama-cpp-vulkan", level="INFO"):
            llama_cpp_unload_model().process("x")
        self.clean.assert_called_once()


if __name__ == "__main__":
    unittest.main()
