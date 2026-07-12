"""app/core/instruct.py 纯函数的单元测试: thinking 块剥离, 预设/自定义提示词组装, 预设配置一致性."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.core.instruct import strip_thinking_blocks, llama_cpp_instruct_base  # noqa: E402
from app.core.prompts import instruct_presets, preset_content  # noqa: E402
from app.nodes.type.text.node_instruct import llama_cpp_text_instruct  # noqa: E402
from app.nodes.type.media.image.node_instruct import llama_cpp_image_instruct  # noqa: E402
from app.nodes.type.media.video.node_instruct import llama_cpp_video_instruct  # noqa: E402
from app.nodes.type.media.audio.node_instruct import llama_cpp_audio_instruct  # noqa: E402

_INSTRUCT_NODES = (
    llama_cpp_text_instruct,
    llama_cpp_image_instruct,
    llama_cpp_video_instruct,
    llama_cpp_audio_instruct,
)


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


class TestBuildUserPrompt(unittest.TestCase):
    """覆盖/填充分支按模板是否含 "###" 判定(预设名的 "(需custom_prompt)" 仅为 UI 提示)。"""

    def setUp(self):
        self.node = llama_cpp_instruct_base()

    def _text(self, preset, custom):
        return self.node._build_user_prompt(preset, custom)["text"]

    def test_plain_preset_overridden_by_custom(self):
        self.assertEqual(self._text("常规 - 描述", "自定义内容"), "自定义内容")

    def test_plain_preset_used_when_custom_empty(self):
        self.assertEqual(self._text("常规 - 描述", ""), "描述这个 图像 。")

    def test_placeholder_preset_fills_custom(self):
        text = self._text("视觉 - BBox 目标检测 (需custom_prompt)", "人物, 车辆")
        self.assertIn('"人物, 车辆"', text)
        self.assertIn("bbox_2d", text)

    def test_placeholder_preset_requires_custom(self):
        with self.assertRaises(ValueError):
            self._text("视觉 - BBox 目标检测 (需custom_prompt)", "  ")

    def test_rewrite_preset_keeps_instruction_and_custom(self):
        # H1 回归: 改写增强预设必须同时保留改写指令与待改写的用户提示词
        text = self._text("创意 - 提示词增强 (需custom_prompt)", "一只猫")
        self.assertIn("改写并增强", text)
        self.assertIn('"一只猫"', text)
        self.assertIn("文生 图像 创作", text)

    def test_rewrite_preset_requires_custom(self):
        with self.assertRaises(ValueError):
            self._text("创意 - 提示词增强 (需custom_prompt)", "")


class TestPresetConfig(unittest.TestCase):
    """预设 use 配置与节点类属性的一致性。"""

    def test_every_node_modality_has_presets(self):
        for node_cls in _INSTRUCT_NODES:
            self.assertTrue(instruct_presets(node_cls.MODALITY))

    def test_default_preset_needs_no_custom_prompt(self):
        # 各模态列表第一项即默认预设, 默认选中就必填 custom_prompt 会造成开箱即报错
        for node_cls in _INSTRUCT_NODES:
            first = instruct_presets(node_cls.MODALITY)[0]
            self.assertNotIn(
                "###", preset_content(first),
                f"{node_cls.__name__} 的默认预设 \"{first}\" 不应要求 custom_prompt",
            )

    def test_all_listed_presets_resolvable(self):
        for node_cls in _INSTRUCT_NODES:
            for name in instruct_presets(node_cls.MODALITY):
                preset_content(name)


if __name__ == "__main__":
    unittest.main()
