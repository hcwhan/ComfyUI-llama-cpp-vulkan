"""src/shared/text_utils.py 的单元测试: 代码围栏剥离, 逐张结果拆分, JSON 解析, 嵌套取值."""

import unittest

from src.shared.text_utils import strip_code_fence, split_image_results, parse_json, get_nested_value


class TestStripCodeFence(unittest.TestCase):
    def test_labeled_fence(self):
        self.assertEqual(strip_code_fence('```json\n{"a": 1}\n```'), '{"a": 1}')

    def test_labeled_fence_with_space_before_label(self):
        # 回归: CommonMark 允许 info string 前有空格, 少数模型输出 "``` json"
        self.assertEqual(strip_code_fence('``` json\n{"a": 1}\n```'), '{"a": 1}')

    def test_leading_prose_block_with_spaced_label(self):
        text = '说明:\n``` json\n{"a": 1}\n```'
        self.assertEqual(strip_code_fence(text), '{"a": 1}')

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

    def test_leading_prose_extracts_first_block(self):
        text = '好的, 结果如下:\n```json\n{"a": 1}\n```'
        self.assertEqual(strip_code_fence(text), '{"a": 1}')

    def test_leading_prose_ignores_trailing_prose(self):
        text = '说明:\n```json\n{"a": 1}\n```\n以上就是结果。'
        self.assertEqual(strip_code_fence(text), '{"a": 1}')

    def test_leading_prose_takes_first_of_multiple_blocks(self):
        text = 'A:\n```\nfirst\n```\nB:\n```\nsecond\n```'
        self.assertEqual(strip_code_fence(text), "first")

    def test_leading_prose_with_unclosed_fence_kept(self):
        # 前导文字 + 未闭合围栏: 无完整块可提取, 保持原样(仅剥尾部围栏的现状行为)
        text = '说明:\n```json\n{"a": 1'
        self.assertEqual(strip_code_fence(text), text)


class TestSplitImageResults(unittest.TestCase):
    def test_multi_image_output_split(self):
        text = "====== Image 1 ======\n\n结果1\n\n====== Image 2 ======\n\n结果2\n\n====== Image 3 ======\n\n结果3"
        self.assertEqual(split_image_results(text), ["结果1", "结果2", "结果3"])

    def test_plain_text_returns_single_element(self):
        self.assertEqual(split_image_results("普通文本"), ["普通文本"])

    def test_json_with_escaped_newline_not_split(self):
        # JSON 文本中的换行是 \n 转义, 不存在真实分隔行, 不应被拆分
        text = '[{"label": "====== Image 1 ======\\n"}]'
        self.assertEqual(split_image_results(text), [text])

    def test_separator_inside_line_not_matched(self):
        # 分隔样式出现在行中间(非独占一行)时不拆分
        text = "前缀 ====== Image 1 ====== 后缀"
        self.assertEqual(split_image_results(text), [text])

    def test_crlf_separator(self):
        text = "====== Image 1 ======\r\n\r\nA\r\n\r\n====== Image 2 ======\r\n\r\nB"
        self.assertEqual(split_image_results(text), ["A", "B"])

    def test_fenced_json_segments(self):
        text = '====== Image 1 ======\n\n```json\n[{"a": 1}]\n```\n\n====== Image 2 ======\n\n```json\n[{"b": 2}]\n```'
        self.assertEqual(split_image_results(text), ['```json\n[{"a": 1}]\n```', '```json\n[{"b": 2}]\n```'])

    def test_leading_text_before_first_separator_kept(self):
        # 首个分隔行之前存在非空正文时保留为首段(防御行为: 实际逐张输出首段恒为空,
        # 但保留语义保证任何输入都不丢内容)
        text = "前导说明\n\n====== Image 1 ======\n\nA"
        self.assertEqual(split_image_results(text), ["前导说明", "A"])

    def test_empty_middle_result_keeps_placeholder(self):
        # 中间某图输出为空时保留空字符串占位, 后续结果不得前移错位
        # (下游 json_to_bboxes / Split Instruct Output 按 "第 i 段对应第 i 张图" 配对)
        text = "====== Image 1 ======\n\nA\n\n====== Image 2 ======\n\n\n\n====== Image 3 ======\n\nC"
        self.assertEqual(split_image_results(text), ["A", "", "C"])

    def test_empty_last_result_keeps_placeholder(self):
        text = "====== Image 1 ======\n\nA\n\n====== Image 2 ======\n\n"
        self.assertEqual(split_image_results(text), ["A", ""])


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
