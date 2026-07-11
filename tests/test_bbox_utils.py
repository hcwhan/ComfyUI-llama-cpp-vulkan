"""app/nodes/type/media/bbox/bbox_utils.py 的单元测试: 坐标换算, 结构校验, 羽化 mask."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.nodes.type.media.bbox.bbox_utils import (  # noqa: E402
    json_to_pixel_bboxes,
    valid_int_bbox,
    feathered_rect_mask,
    bbox_label,
    _label_color,
)


class TestJsonToPixelBboxes(unittest.TestCase):
    def test_simple_mode_passthrough(self):
        items = [{"bbox_2d": [10, 20, 30, 40], "label": "cat"}]
        self.assertEqual(json_to_pixel_bboxes(items, "simple"), [(10, 20, 30, 40)])

    def test_qwen_mode_scales_normalized_coords(self):
        items = [{"bbox_2d": [0, 0, 500, 1000]}]
        result = json_to_pixel_bboxes(items, "Qwen3-VL", width=200, height=100)
        self.assertEqual(result, [(0.0, 0.0, 100.0, 100.0)])

    def test_missing_bbox_2d_raises_value_error(self):
        with self.assertRaises(ValueError):
            json_to_pixel_bboxes([{"label": "cat"}], "simple")

    def test_wrong_length_bbox_raises_value_error(self):
        with self.assertRaises(ValueError):
            json_to_pixel_bboxes([{"bbox_2d": [1, 2, 3]}], "simple")

    def test_non_dict_item_raises_value_error(self):
        with self.assertRaises(ValueError):
            json_to_pixel_bboxes(["not a dict"], "simple")


class TestValidIntBbox(unittest.TestCase):
    def test_rounds_instead_of_truncating(self):
        self.assertEqual(valid_int_bbox((10.6, 0.4, 99.5, 100.0)), (11, 0, 100, 100))

    def test_numeric_strings_accepted(self):
        self.assertEqual(valid_int_bbox(["12.4", "3", "20", "30"]), (12, 3, 20, 30))

    def test_short_sequence_returns_none(self):
        self.assertIsNone(valid_int_bbox([1, 2, 3]))

    def test_non_sequence_returns_none(self):
        self.assertIsNone(valid_int_bbox("nope"))

    def test_non_numeric_value_returns_none(self):
        self.assertIsNone(valid_int_bbox([1, 2, 3, None]))


class TestFeatheredRectMask(unittest.TestCase):
    def test_no_feather_is_binary_rect(self):
        mask = feathered_rect_mask(10, 10, (2, 3, 7, 8), 0)
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask[5, 5], 1.0)
        self.assertEqual(mask[0, 0], 0.0)

    def test_feather_smooths_edges(self):
        mask = feathered_rect_mask(20, 20, (5, 5, 15, 15), 3)
        self.assertEqual(mask.shape, (20, 20))
        # 高斯羽化后应保持 中心 > 矩形边界 > 角落 的单调衰减, 且边界值为中间值
        self.assertGreater(mask[10, 10], mask[5, 10])
        self.assertGreater(mask[5, 10], mask[0, 0])
        self.assertTrue(0.0 < mask[5, 10] < 1.0)
        self.assertLess(mask[0, 0], 0.1)

    def test_empty_rect_stays_zero(self):
        mask = feathered_rect_mask(5, 5, (3, 3, 3, 3), 0)
        self.assertEqual(mask.sum(), 0.0)


class TestLabelHelpers(unittest.TestCase):
    def test_bbox_label_fallbacks(self):
        self.assertEqual(bbox_label({"label": "cat"}), "cat")
        self.assertEqual(bbox_label({"text_content": "dog"}), "dog")
        self.assertEqual(bbox_label({}), "bbox")

    def test_label_color_stable_and_in_range(self):
        c1 = _label_color("cat")
        c2 = _label_color("cat")
        self.assertEqual(c1, c2)
        for ch in c1:
            self.assertTrue(80 <= ch <= 180)


if __name__ == "__main__":
    unittest.main()
