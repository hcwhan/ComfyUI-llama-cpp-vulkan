"""app/nodes/type/media/bbox/bbox_utils.py 的单元测试: 坐标换算, 结构校验, 羽化 mask."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.nodes.type.media.bbox.bbox_utils import (  # noqa: E402
    json_to_pixel_bboxes,
    qwen25_smart_resize,
    valid_int_bbox,
    feathered_rect_mask,
    bbox_label,
    _label_color,
)
from app.nodes.type.media.bbox.node_bbox import SEG  # noqa: E402


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


class TestQwen25SmartResize(unittest.TestCase):
    """M3 回归: smart_resize 复现值须与 Qwen2.5-VL-3B GGUF 的 mtmd 实测一致。

    以下期望值全部来自真实模型推理时 mtmd 日志反推(image chunk token 数),
    并经模型输出坐标逐值印证。
    """

    def test_large_image_downscaled_to_token_cap(self):
        self.assertEqual(qwen25_smart_resize(3200, 2400), (2044, 1540))   # 4015 tok
        self.assertEqual(qwen25_smart_resize(2400, 2400), (1792, 1792))   # 恰 4096 tok

    def test_medium_image_rounded_to_patch_multiple(self):
        self.assertEqual(qwen25_smart_resize(1600, 1200), (1596, 1204))   # 2451 tok
        self.assertEqual(qwen25_smart_resize(800, 600), (812, 588))       # 609 tok

    def test_tiny_image_upscaled_to_min_pixels(self):
        self.assertEqual(qwen25_smart_resize(56, 56), (84, 84))           # 9 tok


class TestQwen25ModeConversion(unittest.TestCase):
    """M3 回归: Qwen2.5-VL 模式把 resize 空间绝对坐标还原为原图坐标。

    用例数据为实测: 3200x2400 原图中方块位于 (2400, 1200)-(2800, 1600),
    模型输出 [1534, 769, 1789, 1025] (resize 后 2044x1540 空间)。
    """

    def test_restores_original_coordinates(self):
        items = [{"bbox_2d": [1534, 769, 1789, 1025], "label": "红色方块"}]
        (box,) = json_to_pixel_bboxes(items, "Qwen2.5-VL", width=3200, height=2400)
        expected = (2400, 1200, 2800, 1600)
        for got, want in zip(box, expected):
            self.assertAlmostEqual(got, want, delta=5)

    def test_identity_scale_below_cap(self):
        # 812x588 (28 对齐后与原图 800x600 略有差异), 坐标按比例微调而非除以 1000
        items = [{"bbox_2d": [609, 294, 711, 392]}]
        (box,) = json_to_pixel_bboxes(items, "Qwen2.5-VL", width=800, height=600)
        expected = (600, 300, 700, 400)
        for got, want in zip(box, expected):
            self.assertAlmostEqual(got, want, delta=1)


class TestSEGNamedtupleCompat(unittest.TestCase):
    """H2 回归: SEG 必须保持 Impact Pack 的 namedtuple 语义。

    Impact Pack 的 SEGSLabelAssign 节点对 SEG 调用 _replace,
    字段名与顺序须与其 modules/impact/core.py 的定义一致。
    """

    def _make_seg(self):
        return SEG(
            cropped_image=None,
            cropped_mask=None,
            confidence=0.9,
            crop_region=[0, 0, 4, 4],
            bbox=(0, 0, 4, 4),
            label="bbox",
        )

    def test_fields_match_impact_pack(self):
        self.assertEqual(
            SEG._fields,
            ("cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label", "control_net_wrapper"),
        )

    def test_control_net_wrapper_defaults_to_none(self):
        self.assertIsNone(self._make_seg().control_net_wrapper)

    def test_replace_relabels_without_touching_other_fields(self):
        relabeled = self._make_seg()._replace(label="person")
        self.assertEqual(relabeled.label, "person")
        self.assertEqual(relabeled.confidence, 0.9)
        self.assertEqual(relabeled.crop_region, [0, 0, 4, 4])


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
