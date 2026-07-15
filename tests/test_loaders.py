"""src/nodes/model/node_loaders.py 的单元测试: loadmodel 快速失败校验分支."""

import re
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import storage  # noqa: E402
from src.core.handlers import HANDLERS  # noqa: E402
from src.i18n.common_static import AUTO_LABEL as _AUTO  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402
from src.nodes.model.node_loaders import (  # noqa: E402
    llama_cpp_llm_model_loader,
    llama_cpp_vlm_model_loader,
)

# 报错文案以语言文件为单一真源, 断言随语言文件自动跟随
_MODEL_NOT_SELECTED = re.escape(LANG["nodes"]["model"]["common"]["errors"]["model_not_selected"])


class TestLlmLoaderValidation(unittest.TestCase):
    def test_model_none_rejected(self):
        with self.assertRaisesRegex(ValueError, _MODEL_NOT_SELECTED):
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
        with self.assertRaisesRegex(ValueError, _MODEL_NOT_SELECTED):
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

    def test_image_tokens_zeroed_for_audio_only_handler(self):
        # 音频专用 handler 无视觉路径, 隐藏字段的残留值折算为 0 落盘,
        # 且不触发 min/max 区间校验 (隐藏字段无法在 UI 修正)
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(handler="(ASR) Qwen3-ASR", min_t=64, max_t=32)
        self.assertEqual(config["image_min_tokens"], 0)
        self.assertEqual(config["image_max_tokens"], 0)

    def test_image_tokens_kept_for_vision_handler(self):
        with mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}"):
            (config,) = self._load(handler="Qwen3-VL", min_t=64, max_t=128)
        self.assertEqual(config["image_min_tokens"], 64)
        self.assertEqual(config["image_max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
