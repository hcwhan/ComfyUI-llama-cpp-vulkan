"""src/core/storage.py resolve_config 的单元测试: loader 快速失败校验的报错路径."""

import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import storage  # noqa: E402
from src.core.handlers import HANDLERS  # noqa: E402


def _config(model="model.gguf", mmproj="None", chat_handler="None"):
    return {"model": model, "mmproj": mmproj, "chat_handler": chat_handler, "thinking": False}


class TestResolveConfig(unittest.TestCase):
    def setUp(self):
        # resolve_config 只做路径与配对校验, 路径查找打桩为恒命中
        patcher = mock.patch.object(storage, "get_llm_full_path", lambda name: f"/fake/{name}")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_model_not_found_raises(self):
        with mock.patch.object(storage, "get_llm_full_path", lambda name: None), self.assertRaises(FileNotFoundError):
            storage.resolve_config(_config())

    def test_unknown_handler_name_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown chat handler"):
            storage.resolve_config(_config(mmproj="m.gguf", chat_handler="No-Such-Handler"))

    def test_mmproj_without_handler_raises(self):
        with self.assertRaisesRegex(ValueError, "chat handler"):
            storage.resolve_config(_config(mmproj="m.gguf", chat_handler="None"))

    def test_handler_without_mmproj_raises(self):
        handler = next(iter(HANDLERS))
        with self.assertRaisesRegex(ValueError, "mmproj"):
            storage.resolve_config(_config(mmproj="None", chat_handler=handler))

    def test_text_only_config_resolves(self):
        model_path, mmproj_path, handler_cls = storage.resolve_config(_config())
        self.assertEqual(model_path, "/fake/model.gguf")
        self.assertIsNone(mmproj_path)
        self.assertIsNone(handler_cls)

    def test_vlm_config_resolves(self):
        handler = next(iter(HANDLERS))
        model_path, mmproj_path, handler_cls = storage.resolve_config(_config(mmproj="m.gguf", chat_handler=handler))
        self.assertEqual(mmproj_path, "/fake/m.gguf")
        self.assertIsNotNone(handler_cls)


if __name__ == "__main__":
    unittest.main()
