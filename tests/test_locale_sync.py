"""src/core/locale_sync.py 的单元测试: 路由注册与 frontend_locale 上报入参校验."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import comfy_stubs

comfy_stubs.install()

from server import PromptServer  # noqa: E402

from src.core import locale_sync  # noqa: E402, F401  (import 即注册路由到替身)
from src.i18n import locale_settings  # noqa: E402


class _FakeRequest:
    def __init__(self, payload=None, raw_text=None, raw_bytes=None, charset=None):
        self._payload = payload
        self._raw_text = raw_text
        self._raw_bytes = raw_bytes
        self._charset = charset

    async def json(self):
        if self._raw_bytes is not None:
            # 复刻 aiohttp json() 经 text() 按 Content-Type charset 解码的路径:
            # 未知 charset 使 bytes.decode 抛 LookupError (非 ValueError 子类)
            return json.loads(self._raw_bytes.decode(self._charset or "utf-8"))
        if self._raw_text is not None:
            return json.loads(self._raw_text)
        return self._payload


class TestFrontendLocaleRoute(unittest.TestCase):
    def setUp(self):
        registered = PromptServer.instance.routes.registered
        matches = [(m, p, h) for m, p, h in registered if p == "/llama_cpp_vulkan/frontend_locale"]
        self.assertEqual(len(matches), 1, "frontend_locale 路由应注册且只注册一次")
        self.assertEqual(matches[0][0], "POST")
        self.handler = matches[0][2]

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "settings.json"
        patcher = patch.object(locale_settings, "_SETTINGS_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, **kwargs):
        return asyncio.run(self.handler(_FakeRequest(**kwargs)))

    def test_valid_locale_persisted(self):
        response = self._post(payload={"locale": "zh"})
        self.assertEqual(response.status, 200)
        self.assertEqual(locale_settings.get_language_setting("frontend_locale"), "zh")

    def test_invalid_json_rejected(self):
        response = self._post(raw_text="{not valid json")
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_bogus_charset_rejected(self):
        # 回归: 请求头伪造未知 charset (application/json; charset=bogus) 时
        # json() 解码抛 LookupError, 修复前不被 except ValueError 捕获, aiohttp
        # 将 handler 异常转为 500; 应视同入参非法返回 400 且不落盘
        response = self._post(raw_bytes=b'{"locale": "zh"}', charset="bogus")
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_non_dict_payload_rejected(self):
        response = self._post(payload=["zh"])
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_missing_locale_rejected(self):
        response = self._post(payload={})
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_non_string_locale_rejected(self):
        response = self._post(payload={"locale": 42})
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_empty_locale_rejected(self):
        response = self._post(payload={"locale": ""})
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())

    def test_overlong_locale_rejected(self):
        response = self._post(payload={"locale": "x" * (locale_sync._MAX_LOCALE_LENGTH + 1)})
        self.assertEqual(response.status, 400)
        self.assertFalse(self.path.is_file())


if __name__ == "__main__":
    unittest.main()
