"""src/core/storage.py resolve_config 的单元测试: loader 快速失败校验的报错路径."""

import re
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

from src.core import storage  # noqa: E402
from src.core.handlers import HANDLERS  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402

# 报错文案以语言文件为单一真源, 断言随语言文件自动跟随
_STORAGE_ERRORS = LANG["common"]["storage_errors"]


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
        expected = re.escape(_STORAGE_ERRORS["unknown_chat_handler"].format(chat_handler="No-Such-Handler"))
        with self.assertRaisesRegex(ValueError, expected):
            storage.resolve_config(_config(mmproj="m.gguf", chat_handler="No-Such-Handler"))

    def test_registered_but_unavailable_handler_raises(self):
        # 注册表声明过但 wheel 缺类的 handler (HANDLERS 缺项抛 KeyError):
        # 报错应指向 "本构建不可用" 而非 "名字未知"
        expected = re.escape(_STORAGE_ERRORS["handler_unavailable"].format(chat_handler="Qwen3-VL"))
        with (
            mock.patch.object(storage, "handler_constructor", side_effect=KeyError),
            mock.patch.object(storage, "is_registered", return_value=True),
            self.assertRaisesRegex(ValueError, expected),
        ):
            storage.resolve_config(_config(mmproj="m.gguf", chat_handler="Qwen3-VL"))

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
