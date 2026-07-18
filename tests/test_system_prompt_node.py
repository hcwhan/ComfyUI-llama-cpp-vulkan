"""src/nodes/util/node_system_prompt.py 的单元测试: "None" 首项语义, 预设查表与模板池非空白."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.util.node_system_prompt import system_prompt_preset  # noqa: E402
from src.nodes.util.system_prompt_presets import PRESETS  # noqa: E402


class TestSystemPromptPreset(unittest.TestCase):
    def setUp(self):
        self.node = system_prompt_preset()

    def test_none_is_first_item_and_default(self):
        decl = system_prompt_preset.INPUT_TYPES()["required"]["preset"]
        choices = decl[0]
        self.assertEqual(choices[0], "None")
        self.assertEqual(choices[1:], list(PRESETS))
        # "None 即默认值" 依赖 ComfyUI "无显式 default 时取首项" 的隐式约定,
        # 锁定声明元组不含带显式 default 的 options 字典
        for options in decl[1:]:
            self.assertNotIn("default", options)

    def test_none_returns_empty_string(self):
        # 空字符串下游 Instruct 不注入 system 消息
        self.assertEqual(self.node.main("None"), ("",))

    def test_none_not_in_presets_pool(self):
        self.assertNotIn("None", PRESETS)

    def test_presets_all_non_blank(self):
        # 若某预设模板为空白, 下游 Instruct 会静默退化为不注入 system 消息, 池级锁定非空白
        for name, template in PRESETS.items():
            with self.subTest(preset=name):
                self.assertTrue(template.strip())

    def test_known_preset_returns_template(self):
        name = next(iter(PRESETS))
        self.assertEqual(self.node.main(name), (PRESETS[name],))

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.node.main("no such preset")


if __name__ == "__main__":
    unittest.main()
