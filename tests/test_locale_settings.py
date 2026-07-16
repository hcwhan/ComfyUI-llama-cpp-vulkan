"""src/i18n/locale_settings.py 的单元测试: settings.json 读写语义 (language 分组, 保留其余内容, 容错)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import comfy_stubs

comfy_stubs.install()

from src.i18n import locale_settings  # noqa: E402


class TestLocaleSettings(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "settings.json"
        patcher = patch.object(locale_settings, "_SETTINGS_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stored(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_set_creates_file_with_language_group(self):
        locale_settings.set_language_setting("frontend_locale", "zh")
        self.assertEqual(self._stored(), {"language": {"frontend_locale": "zh"}})

    def test_get_returns_value_or_none(self):
        self.assertIsNone(locale_settings.get_language_setting("frontend_locale"))
        locale_settings.set_language_setting("frontend_locale", "zh")
        self.assertEqual(locale_settings.get_language_setting("frontend_locale"), "zh")

    def test_set_preserves_other_content(self):
        self.path.write_text(json.dumps({"other": {"a": 1}, "language": {"comfy_locale": "en"}}), encoding="utf-8")
        locale_settings.set_language_setting("frontend_locale", "zh")
        self.assertEqual(self._stored(), {"other": {"a": 1}, "language": {"comfy_locale": "en", "frontend_locale": "zh"}})

    def test_set_none_removes_key(self):
        locale_settings.set_language_setting("comfy_locale", "zh")
        locale_settings.set_language_setting("frontend_locale", "en")
        locale_settings.set_language_setting("comfy_locale", None)
        self.assertEqual(self._stored(), {"language": {"frontend_locale": "en"}})

    def test_unchanged_value_does_not_rewrite(self):
        locale_settings.set_language_setting("frontend_locale", "zh")
        with patch.object(Path, "write_text", side_effect=AssertionError("内容无变化时不应落盘")):
            locale_settings.set_language_setting("frontend_locale", "zh")
            locale_settings.set_language_setting("comfy_locale", None)

    def test_corrupted_file_treated_as_empty(self):
        self.path.write_text("{not valid json", encoding="utf-8")
        self.assertIsNone(locale_settings.get_language_setting("frontend_locale"))
        locale_settings.set_language_setting("frontend_locale", "zh")
        self.assertEqual(self._stored(), {"language": {"frontend_locale": "zh"}})

    def test_non_dict_language_group_treated_as_empty(self):
        self.path.write_text(json.dumps({"language": "zh"}), encoding="utf-8")
        self.assertIsNone(locale_settings.get_language_setting("frontend_locale"))
        locale_settings.set_language_setting("frontend_locale", "zh")
        self.assertEqual(self._stored(), {"language": {"frontend_locale": "zh"}})

    def test_write_failure_is_silent(self):
        with patch.object(locale_settings, "_SETTINGS_PATH", Path(self._tmp.name) / "no_such_dir" / "settings.json"):
            locale_settings.set_language_setting("frontend_locale", "zh")


if __name__ == "__main__":
    unittest.main()
