"""src/nodes/model/node_loaders.py 的单元测试: loadmodel 快速失败校验分支."""

import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import storage  # noqa: E402
from src.core.handlers import HANDLERS  # noqa: E402
from src.nodes.model.node_loaders import (  # noqa: E402
    llama_cpp_llm_model_loader,
    llama_cpp_vlm_model_loader,
)

_AUTO = "Auto (独显优先)"


class TestLlmLoaderValidation(unittest.TestCase):
    def test_model_none_rejected(self):
        with self.assertRaisesRegex(ValueError, "select a gguf model"):
            llama_cpp_llm_model_loader().loadmodel(_AUTO, "None", 8192, -1)

    def test_valid_config_returned(self):
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = llama_cpp_llm_model_loader().loadmodel(_AUTO, "m.gguf", 8192, -1)
        self.assertEqual(config["model"], "m.gguf")
        # llm 侧固定为纯文本配置
        self.assertEqual(config["mmproj"], "None")
        self.assertEqual(config["chat_handler"], "None")


class TestVlmLoaderValidation(unittest.TestCase):
    def _load(self, model="m.gguf", mmproj="mm-mmproj.gguf", handler=None, thinking=False, min_t=0, max_t=0):
        handler = handler or next(iter(HANDLERS))
        return llama_cpp_vlm_model_loader().loadmodel(_AUTO, model, mmproj, handler, thinking, 8192, -1, min_t, max_t)

    def test_model_none_rejected(self):
        with self.assertRaisesRegex(ValueError, "select a gguf model"):
            self._load(model="None")

    def test_mmproj_none_rejected(self):
        with self.assertRaisesRegex(ValueError, "mmproj"):
            self._load(mmproj="None")

    def test_handler_none_rejected(self):
        with self.assertRaisesRegex(ValueError, "chat handler"):
            self._load(handler="None")

    def test_max_tokens_below_min_rejected(self):
        with self.assertRaisesRegex(ValueError, "image_max_tokens"):
            self._load(min_t=64, max_t=32)

    def test_zero_max_tokens_means_unset(self):
        # max=0 视为未设置, 不与 min 做区间比较 (与 handler 侧同一条件)
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(min_t=64, max_t=0)
        self.assertEqual(config["image_min_tokens"], 64)
        self.assertEqual(config["image_max_tokens"], 0)

    def test_thinking_clamped_off_for_unsupported_handler(self):
        # 钳制兜底覆盖绕过前端置灰的路径 (API 提交/旧工作流), 落盘实际生效值
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(handler="Gemma3", thinking=True)
        self.assertFalse(config["thinking"])

    def test_thinking_forced_on_for_thinking_only_handler(self):
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(handler="GLM-4.1V-Thinking", thinking=False)
        self.assertTrue(config["thinking"])

    def test_toggleable_thinking_passes_through(self):
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(handler="Qwen3.6", thinking=True)
        self.assertTrue(config["thinking"])


if __name__ == "__main__":
    unittest.main()
