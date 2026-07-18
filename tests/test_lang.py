"""src/i18n/lang.py 的单元测试: _resolve_language 三级优先级与来源标识, comfy_locale 落盘, 映射表与语言文件的一致性."""

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
        project_settings 为插件根 settings.json 初始 dict (None 表示不建文件);
        解析来源标识另存 self.last_source 供来源断言.
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
                resolved, self.last_source = lang._resolve_language()
            stored = json.loads(project_path.read_text(encoding="utf-8")) if project_path.is_file() else None
            return resolved, stored

    def test_explicit_language_bypasses_settings(self):
        # 非 auto 时原样返回, 不触碰 ComfyUI 设置 (第 1 级) 与插件根
        # settings.json (第 2 级), 读写哨兵任一被调用即失败
        def _fail(*args):
            raise AssertionError("LANGUAGE 非 auto 时不应读写语言设置")

        with (
            patch.object(lang, "LANGUAGE", "zh-CN"),
            patch.object(locale_settings.folder_paths, "get_user_directory", _fail),
            patch.object(locale_settings, "get_language_setting", _fail),
            patch.object(locale_settings, "set_language_setting", _fail),
        ):
            self.assertEqual(lang._resolve_language(), ("zh-CN", "LANGUAGE"))

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

    def test_non_string_locale_falls_through_to_frontend_locale(self):
        # 回归: 旧实现只判 is not None, list 型 Comfy.Locale (设置文件被外力写坏)
        # 查映射表抛 TypeError 阻断插件加载; 现按缺失处理, 继续下一级
        resolved, _ = self._resolve(
            json.dumps({"Comfy.Locale": ["zh"]}),
            project_settings={"language": {"frontend_locale": "zh"}},
        )
        self.assertEqual(resolved, "zh-CN")

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

    def test_non_string_frontend_locale_falls_back_to_default(self):
        # 回归: 旧实现只判 is not None, dict 型 frontend_locale (settings.json 被外力写坏)
        # 查映射表抛 TypeError 阻断插件加载; 现按缺失处理, 落默认英语
        resolved, _ = self._resolve(None, project_settings={"language": {"frontend_locale": {"a": 1}}})
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    # ---- 第 3 级: 默认英语 ----

    def test_nothing_available_falls_back_to_default(self):
        resolved, _ = self._resolve(None)
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    # ---- 来源标识 (启动日志显示) ----

    def test_source_reports_hit_level(self):
        self._resolve(json.dumps({"Comfy.Locale": "zh"}))
        self.assertEqual(self.last_source, "Comfy.Locale")
        self._resolve(None, project_settings={"language": {"frontend_locale": "zh"}})
        self.assertEqual(self.last_source, "frontend_locale")
        self._resolve(None)
        self.assertEqual(self.last_source, "default")

    def test_source_keeps_hit_level_for_unmapped_locale(self):
        # 短码无对应文案时语言落默认英语, 来源仍如实标注提供短码的层级
        # (日志 "en-US (来源: Comfy.Locale)" 解释 "设了 ja 为何显示英语")
        self._resolve(json.dumps({"Comfy.Locale": "ja"}))
        self.assertEqual(self.last_source, "Comfy.Locale")

    def test_corrupted_comfy_settings_falls_back_to_default(self):
        resolved, _ = self._resolve("{not valid json")
        self.assertEqual(resolved, lang._DEFAULT_LANGUAGE)

    def test_non_object_comfy_settings_falls_back_to_default(self):
        # 回归: 旧实现对 json.loads 结果直接 .get, 顶层非对象的合法 JSON (如数组)
        # 抛 AttributeError 阻断插件加载; 现视同键缺失, 落默认英语
        resolved, _ = self._resolve(json.dumps([]))
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
        # 映射表指向的语言文件必须真实存在: 非默认语言缺失运行期打回退
        # warning, 默认语言 (en-US) 缺失直接抛 FileNotFoundError
        for lang_code in set(lang._LOCALE_TO_LANGUAGE.values()):
            with self.subTest(language=lang_code):
                self.assertTrue((lang._I18N_DIR / f"language_{lang_code}.py").is_file())

    def test_stub_user_directory_has_no_settings_file(self):
        # 第 1 级隔离: comfy_stubs 的用户目录必须不存在, Comfy.Locale 确定性走
        # "设置文件缺失" 分支 (与下一用例合并担保 import 期确定性默认英语)
        stub_settings = os.path.join(locale_settings.folder_paths.get_user_directory(), "default", "comfy.settings.json")
        self.assertFalse(os.path.exists(stub_settings))

    def test_stub_settings_path_isolated_from_repo(self):
        # 第 2 级隔离: comfy_stubs 须把插件根 settings.json 重定向出仓库根
        # (junction 部署下该运行时产物真实存在于仓库根, 不隔离则测试进程
        # import 期按其内容解析语言, 且 read_comfy_locale 的落盘收尾会改写
        # 生产文件) 且指向不存在的文件, frontend_locale 确定性走 "文件缺失" 分支
        settings_path = locale_settings._SETTINGS_PATH
        self.assertNotIn(Path(__file__).resolve().parents[1], settings_path.parents)
        self.assertFalse(settings_path.exists())


if __name__ == "__main__":
    unittest.main()
