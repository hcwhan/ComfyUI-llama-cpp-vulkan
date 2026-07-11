"""app/core/instruct.py 纯函数的单元测试: thinking 块剥离."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.core.instruct import strip_thinking_blocks  # noqa: E402


class TestStripThinkingBlocks(unittest.TestCase):
    def test_full_block_removed(self):
        self.assertEqual(strip_thinking_blocks("<think>reasoning</think>answer"), "answer")

    def test_no_block_passthrough(self):
        self.assertEqual(strip_thinking_blocks("plain answer"), "plain answer")

    def test_closing_tag_only(self):
        # generation prompt 已注入开头的 <think>, 输出只含闭合标签
        self.assertEqual(strip_thinking_blocks("reasoning</think>answer"), "answer")

    def test_unclosed_block_kept(self):
        # 生成被截断, 未闭合时保持原样
        text = "<think>truncated reasoning"
        self.assertEqual(strip_thinking_blocks(text), text)

    def test_multiple_blocks(self):
        text = "<think>a</think>mid<think>b</think>end"
        self.assertEqual(strip_thinking_blocks(text), "midend")

    def test_multiline_block(self):
        self.assertEqual(strip_thinking_blocks("<think>line1\nline2</think>\nanswer"), "answer")


if __name__ == "__main__":
    unittest.main()
