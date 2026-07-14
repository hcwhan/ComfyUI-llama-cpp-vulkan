"""src/nodes/util/node_system_prompt.py 的单元测试: "None" 首项语义与预设查表."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.util.node_system_prompt import system_prompt_preset  # noqa: E402
from src.nodes.util.system_prompt_presets import PRESETS  # noqa: E402


class TestSystemPromptPreset(unittest.TestCase):
    def setUp(self):
        self.node = system_prompt_preset()

    def test_none_is_first_item_and_default(self):
        choices = system_prompt_preset.INPUT_TYPES()["required"]["preset"][0]
        self.assertEqual(choices[0], "None")
        self.assertEqual(choices[1:], list(PRESETS))

    def test_none_returns_empty_string(self):
        # 空字符串下游 Instruct 不注入 system 消息
        self.assertEqual(self.node.main("None"), ("",))

    def test_none_not_in_presets_pool(self):
        self.assertNotIn("None", PRESETS)

    def test_known_preset_returns_template(self):
        name = next(iter(PRESETS))
        self.assertEqual(self.node.main(name), (PRESETS[name],))

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.node.main("no such preset")


if __name__ == "__main__":
    unittest.main()
