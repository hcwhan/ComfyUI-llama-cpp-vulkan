"""app/shared/text_utils.py 的单元测试: 代码围栏剥离, JSON 解析, 嵌套取值."""

import unittest

from app.shared.text_utils import strip_code_fence, parse_json, get_nested_value


class TestStripCodeFence(unittest.TestCase):
    def test_labeled_fence(self):
        self.assertEqual(strip_code_fence('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_bare_fence(self):
        self.assertEqual(strip_code_fence("```\nhello\n```"), "hello")

    def test_no_fence_passthrough(self):
        self.assertEqual(strip_code_fence("plain text"), "plain text")

    def test_unclosed_fence_strips_opening_only(self):
        self.assertEqual(strip_code_fence('```json\n{"a": 1'), '{"a": 1')

    def test_fence_with_crlf(self):
        self.assertEqual(strip_code_fence('```json\r\n{"a": 1}\r\n```'), '{"a": 1}')

    def test_label_with_plus_and_dot(self):
        self.assertEqual(strip_code_fence("```c++\ncode\n```"), "code")

    def test_inner_backticks_preserved(self):
        text = "```\nuse `foo` here\n```"
        self.assertEqual(strip_code_fence(text), "use `foo` here")

    def test_surrounding_whitespace(self):
        self.assertEqual(strip_code_fence("  ```\nx\n```  "), "x")


class TestParseJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_json('```json\n[{"b": 2}]\n```'), [{"b": 2}])

    def test_invalid_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_json("not json at all")


class TestGetNestedValue(unittest.TestCase):
    def test_dotted_key(self):
        self.assertEqual(get_nested_value({"a": {"b": 3}}, "a.b"), 3)

    def test_missing_key_returns_default(self):
        self.assertEqual(get_nested_value({"a": 1}, "a.b", "dflt"), "dflt")

    def test_json_in_string_field(self):
        data = {"outer": '{"inner": 7}'}
        self.assertEqual(get_nested_value(data, "outer.inner"), 7)

    def test_unparseable_string_returns_default(self):
        data = {"outer": "not json"}
        self.assertIsNone(get_nested_value(data, "outer.inner"))

    def test_top_level_key(self):
        self.assertEqual(get_nested_value({"k": "v"}, "k"), "v")


if __name__ == "__main__":
    unittest.main()
