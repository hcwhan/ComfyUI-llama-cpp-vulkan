"""src/i18n/lang.py 的单元测试: _resolve_language 三级优先级, comfy_locale 落盘, 映射表与语言文件的一致性."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import comfy_stubs

comfy_stubs.install()

from src.i18n import lang, locale_settings  # noqa: E402


class TestResolveLanguage(unittest.TestCase):
    def _resolve(self, comfy_settings_text=None, project_settings=None):
        """在临时目录中解析语言, 返回 (解析结果, 解析后的项目 settings.json 内容).

        comfy_settings_text 为 default/comfy.settings.json 文本 (None 表示不建文件),
        project_settings 为插件根 settings.json 初始 dict (None 表示不建文件).
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            user_dir = Path(tmp_dir) / "user"
            if comfy_settings_text is not None:
                default_dir = user_dir / "default"
                default_dir.mkdir(parents=True)
                (default_dir / "comfy.settings.json").write_text(comfy_settings_text, encoding="utf-8")
            project_path = Path(tmp_dir) / "settings.json"
            if project_settings is not None:
                project_path.write_text(json.dumps(project_settings), encoding="utf-8")
            with (
                patch.object(lang, "LANGUAGE", "auto"),
                patch.object(locale_settings.folder_paths, "get_user_directory", lambda: str(user_dir)),
                patch.object(locale_settings, "_SETTINGS_PATH", project_path),
            ):
                resolved = lang._resolve_language()
            stored = json.loads(project_path.read_text(encoding="utf-8")) if project_path.is_file() else None
            return resolved, stored

    def test_explicit_language_bypasses_settings(self):
        # 非 auto 时原样返回, 不触碰 ComfyUI 设置 (get_user_directory 被调用即失败)
        def _fail():
            raise AssertionError("LANGUAGE 非 auto 时不应读 ComfyUI 设置")

        with (
            patch.object(lang, "LANGUAGE", "zh-CN"),
            patch.object(locale_settings.folder_paths, "get_user_directory", _fail),
        ):
            self.assertEqual(lang._resolve_language(), "zh-CN")

    # ---- 第 1 级: 实时 Comfy.Locale ----

    def test_locale_zh_maps_to_zh_cn(self):
        resolved, _ = self._resolve(json.dumps({"Comfy.Locale": "zh"}))
        self.assertEqual(resolved, "zh-CN")

    def test_locale_zh_tw_maps_to_zh_cn(self):
        resolved, _ = self._resolve(json.dumps({"Comfy.Locale": "zh-TW"}))
        self.assertEqual(resolved, "zh-CN")

    def test_locale_en_maps_to_en_us(self):
        resolved, _ = self._resolve(json.dumps({"Comfy.Locale": "en"}))
        self.assertEqual(resolved, "en-US")

    def test_unmapped_locale_falls_back_to_default(self):
        resolved, _ = self._resolve(json.dumps({"Comfy.Locale": "ja"}))
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    def test_comfy_locale_beats_frontend_locale(self):
        resolved, _ = self._resolve(
            json.dumps({"Comfy.Locale": "en"}),
            project_settings={"language": {"frontend_locale": "zh"}},
        )
        self.assertEqual(resolved, "en-US")

    # ---- 第 2 级: 上次会话前端上报 ----

    def test_frontend_locale_fallback_when_comfy_locale_missing(self):
        resolved, _ = self._resolve(
            json.dumps({"Comfy.UseNewMenu": "Top"}),
            project_settings={"language": {"frontend_locale": "zh"}},
        )
        self.assertEqual(resolved, "zh-CN")

    def test_frontend_locale_fallback_when_settings_file_missing(self):
        resolved, _ = self._resolve(None, project_settings={"language": {"frontend_locale": "zh-TW"}})
        self.assertEqual(resolved, "zh-CN")

    def test_unmapped_frontend_locale_falls_back_to_default(self):
        resolved, _ = self._resolve(None, project_settings={"language": {"frontend_locale": "ja"}})
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    # ---- 第 3 级: 默认英语 ----

    def test_nothing_available_falls_back_to_default(self):
        resolved, _ = self._resolve(None)
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    def test_corrupted_comfy_settings_falls_back_to_default(self):
        resolved, _ = self._resolve("{not valid json")
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    # ---- comfy_locale 落盘 (忠实记录) ----

    def test_comfy_locale_recorded_on_resolve(self):
        _, stored = self._resolve(json.dumps({"Comfy.Locale": "zh"}))
        self.assertEqual(stored["language"]["comfy_locale"], "zh")

    def test_comfy_locale_removed_when_missing(self):
        _, stored = self._resolve(
            json.dumps({"Comfy.UseNewMenu": "Top"}),
            project_settings={"language": {"comfy_locale": "zh", "frontend_locale": "zh"}},
        )
        self.assertEqual(stored["language"], {"frontend_locale": "zh"})

    def test_no_project_file_created_when_nothing_to_record(self):
        _, stored = self._resolve(None)
        self.assertIsNone(stored)


class TestLocaleMappingContract(unittest.TestCase):
    def test_mapping_targets_have_language_files(self):
        # 映射表指向的语言文件必须真实存在, 否则运行期会打回退 warning
        for lang_code in set(lang._LOCALE_TO_LANGUAGE.values()):
            with self.subTest(language=lang_code):
                self.assertTrue((lang._I18N_DIR / f"language_{lang_code}.py").is_file())

    def test_stub_user_directory_has_no_settings_file(self):
        # comfy_stubs 的用户目录必须不存在, 保证其他测试模块 import 期加载确定性的默认英语
        stub_settings = os.path.join(locale_settings.folder_paths.get_user_directory(), "default", "comfy.settings.json")
        self.assertFalse(os.path.exists(stub_settings))


if __name__ == "__main__":
    unittest.main()
