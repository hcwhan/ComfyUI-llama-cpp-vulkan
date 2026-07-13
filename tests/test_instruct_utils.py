"""src/core/instruct.py 纯函数的单元测试: thinking 块剥离, 预设/自定义提示词组装, 预设配置一致性."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.core.instruct import strip_thinking_blocks, llama_cpp_instruct_base  # noqa: E402
from src.core.prompts import instruct_presets, preset_content  # noqa: E402
from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.nodes.type.text.node_instruct import llama_cpp_text_instruct  # noqa: E402
from src.nodes.type.media.image.node_instruct import llama_cpp_image_instruct  # noqa: E402
from src.nodes.type.media.video.node_instruct import llama_cpp_video_instruct  # noqa: E402
from src.nodes.type.media.audio.node_instruct import llama_cpp_audio_instruct  # noqa: E402

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

    def test_gemma4_channel_block_removed(self):
        # Gemma4 思考块格式: <|channel>thought ... <channel|>正文
        text = "<|channel>thought\nreasoning here<channel|>正文"
        self.assertEqual(strip_thinking_blocks(text), "正文")

    def test_gemma4_channel_without_opening_tag(self):
        # E2B/E4B 在 enable_thinking=False 时仍以纯文本思考并自行输出
        # <channel|> 分隔符(无开标签, 实测确认), 取最后一段
        text = "The user wants the main color.\nIt is red.<channel|>红色"
        self.assertEqual(strip_thinking_blocks(text), "红色")

    def test_gemma4_multiple_channel_marks_take_last(self):
        self.assertEqual(strip_thinking_blocks("a<channel|>b<channel|>c"), "c")

    def test_gemma4_unclosed_thought_kept(self):
        # 生成截断在思考块内部, 无闭合 token 时保持原样(与 <think> 约定一致)
        text = "<|channel>thought\ntruncated reasoning"
        self.assertEqual(strip_thinking_blocks(text), text)

    def test_glm41v_answer_wrapper_removed(self):
        # 回归: GLM-4.1V 输出 <think>...</think>\n<answer>正文</answer>,
        # handler 以 </answer> 为 stop token, 开标签会残留
        self.assertEqual(strip_thinking_blocks("<think>r</think>\n<answer>正文"), "正文")

    def test_glm41v_closed_answer_wrapper_removed(self):
        self.assertEqual(strip_thinking_blocks("<think>r</think>\n<answer>正文</answer>"), "正文")

    def test_answer_wrapper_without_think_block(self):
        # generation prompt 已注入 <think> 时输出只含闭合标签, 同样带 answer 包裹
        self.assertEqual(strip_thinking_blocks("r</think>\n<answer>正文"), "正文")

    def test_answer_word_in_body_untouched(self):
        # 正文中出现 <answer> 字样但不在开头, 不剥离
        text = "标签 <answer> 的用法说明"
        self.assertEqual(strip_thinking_blocks(text), text)


class TestBuildUserPrompt(unittest.TestCase):
    """覆盖/填充分支按模板是否含 "###" 判定(预设名的 "(需custom_prompt)" 仅为 UI 提示)."""

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
        # 回归: 改写增强预设必须同时保留改写指令与待改写的用户提示词
        text = self._text("创意 - 提示词增强 (需custom_prompt)", "一只猫")
        self.assertIn("改写并增强", text)
        self.assertIn('"一只猫"', text)
        self.assertIn("文生 图像 创作", text)

    def test_rewrite_preset_requires_custom(self):
        with self.assertRaises(ValueError):
            self._text("创意 - 提示词增强 (需custom_prompt)", "")


class TestSingleCompletionContentFlattening(unittest.TestCase):
    """回归: 纯文本 user content 必须扁平化为字符串.

    无 chat handler 的文本路径由 GGUF 内嵌 chat template 渲染消息,
    旧式模板(ChatML/Llama-3/Mistral 等)假定 content 是字符串,
    收到 content-part 列表会 TypeError 或渲染出 Python repr 垃圾.
    """

    class _FakeLlm:
        def __init__(self):
            self.captured_messages = None

        def create_chat_completion(self, messages, seed, **params):
            self.captured_messages = messages
            return {"choices": [{"message": {"content": "ok"}}]}

    def setUp(self):
        self.node = llama_cpp_instruct_base()
        self.fake = self._FakeLlm()
        self._orig_llm = LLAMA_CPP_STORAGE.llm
        LLAMA_CPP_STORAGE.llm = self.fake
        self.addCleanup(setattr, LLAMA_CPP_STORAGE, "llm", self._orig_llm)

    @staticmethod
    def _extract(output):
        return output["choices"][0]["message"]["content"]

    def test_single_text_item_flattened_to_string(self):
        self.node._single_completion([], [{"type": "text", "text": "hello"}], 0, {}, self._extract)
        self.assertEqual(self.fake.captured_messages[-1]["content"], "hello")

    def test_media_content_list_passed_through(self):
        content = [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]
        self.node._single_completion([], content, 0, {}, self._extract)
        self.assertIs(self.fake.captured_messages[-1]["content"], content)


class TestRequireUserText(unittest.TestCase):
    """text 路径空 user 文本拦截: 无媒体载荷的空请求在模型加载前报错.

    media 路径不拦截(空文本 + 媒体内容是有意设计).
    """

    def test_text_node_requires_user_text(self):
        self.assertTrue(llama_cpp_text_instruct.REQUIRE_USER_TEXT)

    def test_media_nodes_allow_empty_user_text(self):
        for node_cls in (llama_cpp_image_instruct, llama_cpp_video_instruct, llama_cpp_audio_instruct):
            self.assertFalse(node_cls.REQUIRE_USER_TEXT, node_cls.__name__)

    def test_blank_preset_empty_custom_rejected_before_model_load(self):
        # runner 不应被调用: 拦截必须发生在 _prepare_messages(触发加载)之前
        node = llama_cpp_text_instruct()
        with self.assertRaises(ValueError):
            node._run(
                llama_model={}, preset_prompt="空白 - 空", custom_prompt="  ",
                system_prompt="", seed=0, force_offload=False, strip_thinking=True,
                parameters=None, runner=lambda *args: self.fail("runner should not run"),
            )

    def test_non_empty_custom_passes_guard(self):
        # 守卫放行后才会走到 _prepare_messages, 用空配置 dict 缺 "model" 键
        # 触发的 KeyError 佐证已越过守卫(不实际加载模型)
        node = llama_cpp_text_instruct()
        with self.assertRaises(KeyError):
            node._run(
                llama_model={}, preset_prompt="空白 - 空", custom_prompt="一只猫",
                system_prompt="", seed=0, force_offload=False, strip_thinking=True,
                parameters=None, runner=lambda *args: self.fail("runner should not run"),
            )


class TestPresetConfig(unittest.TestCase):
    """预设 use 配置与节点类属性的一致性."""

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
