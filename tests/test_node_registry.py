"""src/nodes 注册表契约的单元测试: display_names 键一致性, 全节点 INPUT_TYPES() smoke test."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.i18n.lang import LANG  # noqa: E402
from src.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402


class TestNodeRegistry(unittest.TestCase):
    def test_display_names_cover_all_registered_nodes(self):
        # 新增节点漏写显示名会静默回退为类名 (ComfyUI 对缺失键的默认行为),
        # 多写的显示名键则是语言文件的死条目, 两个方向都在此拦截
        self.assertEqual(set(NODE_CLASS_MAPPINGS), set(LANG["display_names"]))

    def test_display_name_mappings_mirror_language_file(self):
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS, LANG["display_names"])

    def test_all_nodes_input_types_callable(self):
        # smoke test: INPUT_TYPES 内部的 LANG 键漏写/改名在 import 期不报错,
        # 要到前端请求 /object_info 时才 KeyError; 全量调用一遍在测试期拦截,
        # 一并锁定 parameters 节点 tooltip 的 {default} 填充路径
        for name, node_cls in NODE_CLASS_MAPPINGS.items():
            with self.subTest(node=name):
                input_types = node_cls.INPUT_TYPES()
                self.assertIsInstance(input_types, dict)
                self.assertIn("required", input_types)


if __name__ == "__main__":
    unittest.main()
