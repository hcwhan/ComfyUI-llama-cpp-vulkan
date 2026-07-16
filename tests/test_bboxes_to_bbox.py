"""src/nodes/bbox/node_bboxes_to_bbox.py 的节点级单元测试.

覆盖: 二级索引正常选取, 负索引 (下界 -len(group), 与 widget min: -998 的
不对称是存档的设计权衡), 999 哨兵返回整组, 两级越界报错 (断言引用 LANG
模板), INPUT_IS_LIST 下 widget 参数的列表解包.
"""

import re
import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.i18n.lang import LANG  # noqa: E402
from src.nodes.bbox.node_bboxes_to_bbox import bboxes_to_bbox  # noqa: E402

_ERRORS = LANG["nodes"]["bbox"]["bboxes_to_bbox"]["errors"]

# 两组 BBox: 图 0 有三个框, 图 1 有一个框
_GROUPS = [
    [(1, 1, 4, 4), (2, 2, 5, 5), (3, 3, 6, 6)],
    [(7, 7, 9, 9)],
]


class TestBBoxesToBBox(unittest.TestCase):
    def _process(self, bboxes, image_index, bbox_index):
        # INPUT_IS_LIST 下 widget 参数被 ComfyUI 包成列表, 传参形态与运行时一致
        return bboxes_to_bbox().process(bboxes, [image_index], [bbox_index])[0]

    def test_selects_by_two_level_index(self):
        self.assertEqual(self._process(_GROUPS, 0, 1), [(2, 2, 5, 5)])
        self.assertEqual(self._process(_GROUPS, 1, 0), [(7, 7, 9, 9)])

    def test_negative_index_counts_from_tail(self):
        self.assertEqual(self._process(_GROUPS, 0, -1), [(3, 3, 6, 6)])
        # 下界 -len(group): 与正索引覆盖同一元素范围
        self.assertEqual(self._process(_GROUPS, 0, -3), [(1, 1, 4, 4)])

    def test_999_sentinel_returns_whole_group(self):
        # 999 哨兵返回整组 (不再包一层), 组本身就是 BBOX 列表
        self.assertEqual(self._process(_GROUPS, 0, 999), _GROUPS[0])

    def test_image_index_out_of_range_raises(self):
        expected = re.escape(_ERRORS["image_index_out_of_range"].format(image_index=2, count=2))
        with self.assertRaisesRegex(IndexError, expected):
            self._process(_GROUPS, 2, 0)

    def test_bbox_index_out_of_range_raises(self):
        expected = re.escape(_ERRORS["bbox_index_out_of_range"].format(bbox_index=3, image_index=0, count=3))
        with self.assertRaisesRegex(IndexError, expected):
            self._process(_GROUPS, 0, 3)

    def test_negative_index_below_lower_bound_raises(self):
        # 负索引下界是 -len(group), widget 的 min: -998 只是 UI 下限,
        # 越过实际组长仍须报错而非绕回
        expected = re.escape(_ERRORS["bbox_index_out_of_range"].format(bbox_index=-4, image_index=0, count=3))
        with self.assertRaisesRegex(IndexError, expected):
            self._process(_GROUPS, 0, -4)

    def test_empty_group_any_index_raises(self):
        # 空组时除 999 外任何索引都越界 (0 也不例外)
        with self.assertRaises(IndexError):
            self._process([[]], 0, 0)
        self.assertEqual(self._process([[]], 0, 999), [])


if __name__ == "__main__":
    unittest.main()
