"""src/nodes/bbox/node_json_to_bboxes.py 的单元测试: JSON 条数与帧数不匹配时的重组对齐."""

import re
import unittest

import torch

from tests import comfy_stubs

comfy_stubs.install()

from src.i18n.common_static import BBOX_MODE_QWEN3, BBOX_MODE_SIMPLE  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402
from src.nodes.bbox.node_json_to_bboxes import json_to_bboxes  # noqa: E402
from src.shared.text_utils import parse_json  # noqa: E402

_JSON_ONE_BOX = '[{"bbox_2d": [1, 1, 4, 4], "label": "a"}]'


def _frames(batch_sizes, size=16):
    return [torch.zeros((n, size, size, 3), dtype=torch.float32) for n in batch_sizes]


class TestJsonToBBoxesRestructure(unittest.TestCase):
    def setUp(self):
        self.node = json_to_bboxes()

    def _process(self, n_json, batch_sizes):
        return self.node.process([_JSON_ONE_BOX] * n_json, [BBOX_MODE_SIMPLE], [""], _frames(batch_sizes))

    def test_matched_counts_keep_batch_structure(self):
        bboxes, image_list = self._process(4, [2, 2])
        self.assertEqual(len(bboxes), 4)
        self.assertEqual([b.shape[0] for b in image_list], [2, 2])

    def test_fewer_json_passes_unpaired_frames_through(self):
        # 回归: JSON 少于帧时未配对的帧原样进入输出(不画框), 批次结构不塌陷
        bboxes, image_list = self._process(3, [2, 2])
        self.assertEqual(len(bboxes), 3)
        self.assertEqual([b.shape[0] for b in image_list], [2, 2])

    def test_single_json_many_frames(self):
        bboxes, image_list = self._process(1, [2, 2])
        self.assertEqual(len(bboxes), 1)
        self.assertEqual([b.shape[0] for b in image_list], [2, 2])

    def test_extra_json_appended_as_single_frame_batches(self):
        # 回归: JSON 多于帧时多画的帧作为单帧批次追加, 总帧数与 bboxes 组数对齐
        bboxes, image_list = self._process(4, [2])
        self.assertEqual(len(bboxes), 4)
        self.assertEqual([b.shape[0] for b in image_list], [2, 1, 1])

    def test_no_images_returns_empty_image_list(self):
        bboxes, image_list = self.node.process([_JSON_ONE_BOX], [BBOX_MODE_SIMPLE], [""], None)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(image_list, [])

    def test_parse_error_reports_segment_index(self):
        # 回归: 逐段解析失败的报错须带分段索引 (从 1 起, 与前缀行 Image N 对齐),
        # 便于定位坏在哪张图的输出; 坏段是第 2 段, 报错应为 JSON #2.
        # {error} 实参取 parse_json 对同一坏段的真实报错, 使全文案精确断言
        with self.assertRaises(ValueError) as inner:
            parse_json("not json")
        expected = re.escape(LANG["nodes"]["bbox"]["json_to_bboxes"]["errors"]["json_parse_failed"].format(i=2, error=inner.exception))
        with self.assertRaisesRegex(ValueError, expected):
            self.node.process([_JSON_ONE_BOX, "not json"], [BBOX_MODE_SIMPLE], [""], None)


_JSON_LABELED = (
    '[{"bbox_2d": [1, 1, 4, 4], "label": "Cat"},'
    ' {"bbox_2d": [2, 2, 5, 5], "label": "dog"},'
    ' {"bbox_2d": [3, 3, 6, 6], "text_content": " cat "}]'
)


class TestJsonToBBoxesLabelFilter(unittest.TestCase):
    def setUp(self):
        self.node = json_to_bboxes()

    def test_filter_ignores_case_and_matches_text_content(self):
        # label 匹配忽略大小写与首尾空格, label / text_content 任一字段命中即保留
        bboxes, _ = self.node.process([_JSON_LABELED], [BBOX_MODE_SIMPLE], ["cat"], None)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [(1, 1, 4, 4), (3, 3, 6, 6)])

    def test_empty_label_keeps_all(self):
        bboxes, _ = self.node.process([_JSON_LABELED], [BBOX_MODE_SIMPLE], [""], None)
        self.assertEqual(len(bboxes[0]), 3)

    def test_no_match_returns_empty_group(self):
        bboxes, _ = self.node.process([_JSON_LABELED], [BBOX_MODE_SIMPLE], ["bird"], None)
        self.assertEqual(bboxes[0], [])

    def test_non_dict_item_with_filter_reports_structure_error(self):
        # 回归: label 过滤开启时非 dict 项(如坐标数组)不得抛裸 AttributeError,
        # 应留给结构校验报出带期望格式的 ValueError
        with self.assertRaises(ValueError):
            self.node.process(["[[10, 20, 30, 40]]"], [BBOX_MODE_SIMPLE], ["cat"], None)

    def test_numeric_label_matched_by_string_filter(self):
        # 回归: LLM 输出数字标签时画框显示 "5" (bbox_label 强转 str),
        # 过滤框填 "5" 须能匹配到该框, 匹配与显示路径行为一致
        json_str = '[{"bbox_2d": [1, 1, 4, 4], "label": 5}, {"bbox_2d": [2, 2, 5, 5], "label": "cat"}]'
        bboxes, _ = self.node.process([json_str], [BBOX_MODE_SIMPLE], ["5"], None)
        self.assertEqual(bboxes[0], [(1, 1, 4, 4)])

    def test_missing_label_fields_matched_by_fallback(self):
        # 回归: 无 label/text_content 字段的项画框显示 fallback 标签 "bbox",
        # 过滤框填 "bbox" 须能匹配到该项 (与 bbox_label 显示路径同源取值);
        # 填其他标签时缺字段项不得被误保留
        json_str = '[{"bbox_2d": [1, 1, 4, 4]}, {"bbox_2d": [2, 2, 5, 5], "label": "cat"}]'
        bboxes, _ = self.node.process([json_str], [BBOX_MODE_SIMPLE], ["bbox"], None)
        self.assertEqual(bboxes[0], [(1, 1, 4, 4)])
        bboxes, _ = self.node.process([json_str], [BBOX_MODE_SIMPLE], ["cat"], None)
        self.assertEqual(bboxes[0], [(2, 2, 5, 5)])


class TestJsonToBBoxesQwenMode(unittest.TestCase):
    def setUp(self):
        self.node = json_to_bboxes()

    def test_qwen_mode_requires_image(self):
        # Qwen 坐标系换算依赖原图尺寸, 未连 image 时须明确报错
        expected = re.escape(LANG["nodes"]["bbox"]["json_to_bboxes"]["errors"]["image_required"])
        with self.assertRaisesRegex(ValueError, expected):
            self.node.process(['[{"bbox_2d": [0, 0, 500, 1000]}]'], [BBOX_MODE_QWEN3], [""], None)

    def test_qwen3_normalized_coords_scaled_to_frame(self):
        # Qwen3-VL 输出 0-1000 归一化坐标, 按帧尺寸 (w=200, h=100) 换算
        frames = [torch.zeros(1, 100, 200, 3)]
        bboxes, image_list = self.node.process(['[{"bbox_2d": [0, 0, 500, 1000], "label": "x"}]'], [BBOX_MODE_QWEN3], [""], frames)
        self.assertEqual(bboxes[0], [(0.0, 0.0, 100.0, 100.0)])
        self.assertEqual(tuple(image_list[0].shape), (1, 100, 200, 3))


if __name__ == "__main__":
    unittest.main()
