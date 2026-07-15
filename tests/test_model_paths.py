"""src/core/model_paths.py 的单元测试: 双键去重, gguf 过滤, 路径查找, 扩展名集合重建与共享集合防污染."""

import importlib
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

import folder_paths  # noqa: E402

from src.core import model_paths  # noqa: E402


class TestGetLlmFilenameList(unittest.TestCase):
    def test_dedup_across_keys_and_gguf_filter(self):
        # llm/LLM 双键结果去重, 其他插件追加的非 gguf 扩展名被过滤
        listing = {"llm": ["a.gguf", "b.txt", "c.GGUF"], "LLM": ["a.gguf", "d.gguf"]}
        with mock.patch.object(folder_paths, "get_filename_list", lambda key: listing[key]):
            result = model_paths.get_llm_filename_list()
        self.assertEqual(result, ["a.gguf", "c.GGUF", "d.gguf"])

    def test_merged_result_globally_sorted(self):
        # 两键为独立目录时 (Linux 下 llm/LLM 大小写不同名), 合并结果须全局
        # 排序而非按键拼接, 排序规则与 folder_paths.get_filename_list 的
        # 单键结果一致 (朴素 sorted, 无大小写折叠)
        listing = {"llm": ["z.gguf", "m.gguf"], "LLM": ["a.gguf"]}
        with mock.patch.object(folder_paths, "get_filename_list", lambda key: listing[key]):
            result = model_paths.get_llm_filename_list()
        self.assertEqual(result, ["a.gguf", "m.gguf", "z.gguf"])

    def test_full_path_falls_through_keys(self):
        # 首键未命中时继续查后续目录键
        def fake_full(key, name):
            return f"/{key}/{name}" if key == "LLM" else None

        with mock.patch.object(folder_paths, "get_full_path", fake_full):
            self.assertEqual(model_paths.get_llm_full_path("x.gguf"), "/LLM/x.gguf")

    def test_missing_file_returns_none(self):
        with mock.patch.object(folder_paths, "get_full_path", lambda key, name: None):
            self.assertIsNone(model_paths.get_llm_full_path("x.gguf"))


class TestExtensionSetRebuild(unittest.TestCase):
    def test_list_extensions_rebuilt_as_set(self):
        # 键已被其他插件以 (paths, list) 形态注册时, import 期须重建为 set
        # 并合并 .gguf, 避免 AttributeError 使整个插件加载失败
        original = dict(folder_paths.folder_names_and_paths)
        self.addCleanup(folder_paths.folder_names_and_paths.update, original)
        folder_paths.folder_names_and_paths["llm"] = (["/x"], [".bin"])
        folder_paths.folder_names_and_paths["LLM"] = (["/x"], {".gguf"})

        importlib.reload(model_paths)

        _paths, exts = folder_paths.folder_names_and_paths["llm"]
        self.assertIsInstance(exts, set)
        self.assertEqual(exts, {".bin", ".gguf"})

    def test_shared_extension_set_not_polluted(self):
        # 回归: 第三方插件注册 llm/LLM 键时可能直接引用与内置键共享的扩展名
        # 集合 (最典型是 supported_pt_extensions, ComfyUI 约 20 个内置键共用
        # 同一 set 实例). 本插件须复制重建而非原位 update, 否则 .gguf 会
        # 泄漏进 checkpoints/loras 等全部内置下拉框
        original = dict(folder_paths.folder_names_and_paths)
        self.addCleanup(folder_paths.folder_names_and_paths.update, original)
        shared = {".safetensors"}
        folder_paths.folder_names_and_paths["checkpoints"] = (["/ckpt"], shared)
        folder_paths.folder_names_and_paths["llm"] = (["/x"], shared)
        folder_paths.folder_names_and_paths["LLM"] = (["/x"], shared)

        importlib.reload(model_paths)

        # 共享集合原封不动, 只有 llm/LLM 两键的新集合多出 .gguf
        self.assertEqual(shared, {".safetensors"})
        self.assertEqual(folder_paths.folder_names_and_paths["checkpoints"][1], {".safetensors"})
        for key in ("llm", "LLM"):
            self.assertEqual(folder_paths.folder_names_and_paths[key][1], {".safetensors", ".gguf"})


if __name__ == "__main__":
    unittest.main()
