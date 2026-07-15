"""src/nodes 注册表契约的单元测试: NODE_CLASS_MAPPINGS 与语言文件 display_names 的键一致性."""

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


if __name__ == "__main__":
    unittest.main()
