"""src/i18n 语言文件排版约定与跨语言一致性的单元测试.

锁定 language_*.py 的排版规则 (见语言文件 docstring 的排版约定):
- 多字面量拼接的非末位字面量必须以 \\n 结尾: 保证源码行与 UI 显示行一一对应,
  漏写 \\n 会把两个显示行粘连成一行
- 多字面量拼接的末位字面量不能以 \\n 结尾: UI 文本不带尾随空行
- errors 类分组 (键名以 errors 结尾) 的文案必须是单行

前两条基于 tokenize 做源码级检查, 不 import 语言文件, 与运行时行为解耦;
第三条按路径加载 LANG dict 检查 (纯数据文件, 无第三方依赖).

另锁定多语言文件间的结构契约 (AGENTS.md 的 "逐行一一对应" 约定):
- 全部语言文件的 LANG 叶子键路径列表 (含顺序) 完全一致
- 同一叶子键的 str.format 具名占位符名称集合完全一致
"""

import ast
import importlib.util
import io
import string
import tokenize
import unittest
from pathlib import Path

_I18N_DIR = Path(__file__).resolve().parent.parent / "src" / "i18n"

# 隐式拼接串内部允许出现的 token 类型 (换行/注释/缩进不打断相邻字面量拼接)
_JOINABLE_TYPES = (tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT)


def _language_files():
    return sorted(_I18N_DIR.glob("language_*.py"))


def _load_lang(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LANG


def _iter_error_entries(node, path=()):
    """遍历 LANG dict, 产出 errors 类分组 (键名以 errors 结尾) 下的全部叶子文案."""
    for key, value in node.items():
        sub_path = (*path, key)
        if isinstance(value, dict):
            yield from _iter_error_entries(value, sub_path)
        elif any(seg.endswith("errors") for seg in path):
            yield sub_path, value


def _flatten_leaves(node, path=()):
    """按声明顺序递归展平 LANG dict, 产出 (叶子键路径元组, 叶子值)."""
    for key, value in node.items():
        sub_path = (*path, key)
        if isinstance(value, dict):
            yield from _flatten_leaves(value, sub_path)
        else:
            yield sub_path, value


def _placeholder_names(text):
    """提取 str.format 具名占位符名称集合 ({{ }} 转义与空自动编号不计入)."""
    return {field for _, field, _, _ in string.Formatter().parse(text) if field}


def _concat_runs(source):
    """返回源码中全部隐式拼接串, 每串为相邻 STRING token 列表 (长度 >= 2).

    模块 docstring 与普通单字面量是长度 1 的串, 不属于多字面量形态, 被过滤;
    dict 的 key/value 字符串之间隔着 ':' 等 OP token, 不会被误并为一串.
    """
    runs = []
    current = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.STRING:
            current.append(tok)
        elif tok.type in _JOINABLE_TYPES:
            continue
        else:
            if len(current) > 1:
                runs.append(current)
            current = []
    if len(current) > 1:
        runs.append(current)
    return runs


class TestLanguageFileLayout(unittest.TestCase):
    def test_language_files_exist(self):
        # 目录结构变动导致 glob 落空时, 三条逐文件排版测试会静默变成空跑
        # (循环体不执行), 在此拦截; 两条跨语言测试对空集是解包报错, 不静默
        self.assertTrue(_language_files(), f"{_I18N_DIR} 下找不到 language_*.py")

    def test_non_last_literals_end_with_newline(self):
        for path in _language_files():
            runs = _concat_runs(path.read_text(encoding="utf-8"))
            self.assertTrue(runs, f"{path.name} 中没有多字面量拼接串, 检查是否被整体改写")
            for run in runs:
                for tok in run[:-1]:
                    with self.subTest(file=path.name, line=tok.start[0]):
                        self.assertTrue(
                            ast.literal_eval(tok.string).endswith("\n"),
                            f"{path.name}:{tok.start[0]} 非末位字面量必须以 \\n 结尾: {tok.string}",
                        )

    def test_last_literal_not_end_with_newline(self):
        for path in _language_files():
            for run in _concat_runs(path.read_text(encoding="utf-8")):
                tok = run[-1]
                with self.subTest(file=path.name, line=tok.start[0]):
                    self.assertFalse(
                        ast.literal_eval(tok.string).endswith("\n"),
                        f"{path.name}:{tok.start[0]} 末位字面量不能以 \\n 结尾: {tok.string}",
                    )

    def test_error_entries_are_single_line(self):
        for path in _language_files():
            entries = list(_iter_error_entries(_load_lang(path)))
            self.assertTrue(entries, f"{path.name} 中找不到 errors 类分组, 检查是否被整体改写")
            for key_path, text in entries:
                with self.subTest(file=path.name, key=".".join(key_path)):
                    self.assertNotIn(
                        "\n",
                        text,
                        f"{path.name} 的 {'.'.join(key_path)} 是报错文案, 必须为单行: {text!r}",
                    )


class TestLanguageFilesConsistency(unittest.TestCase):
    """多语言文件间的键结构契约: 漏更新一侧时在此拦截 (默认 zh-CN 下 en-US 的缺键运行期永不触发)."""

    def _leaves_by_file(self):
        return {path.name: list(_flatten_leaves(_load_lang(path))) for path in _language_files()}

    def test_leaf_key_paths_identical_in_order(self):
        leaves = self._leaves_by_file()
        base_name, *other_names = leaves
        base_keys = [key_path for key_path, _ in leaves[base_name]]
        for name in other_names:
            with self.subTest(file=name):
                self.assertEqual(
                    base_keys,
                    [key_path for key_path, _ in leaves[name]],
                    f"{name} 与 {base_name} 的叶子键路径 (含顺序) 不一致",
                )

    def test_leaf_placeholder_names_identical(self):
        leaves = self._leaves_by_file()
        base_name, *other_names = leaves
        base_placeholders = {key_path: _placeholder_names(value) for key_path, value in leaves[base_name]}
        for name in other_names:
            for key_path, value in leaves[name]:
                with self.subTest(file=name, key=".".join(key_path)):
                    self.assertEqual(
                        base_placeholders.get(key_path),
                        _placeholder_names(value),
                        f"{name} 的 {'.'.join(key_path)} 占位符与 {base_name} 不一致",
                    )


if __name__ == "__main__":
    unittest.main()
