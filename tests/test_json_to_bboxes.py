"""src/nodes/type/media/bbox/node_bbox.py json_to_bboxes 的单元测试: JSON 条数与帧数不匹配时的重组对齐."""

import unittest

import torch

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.type.media.bbox.node_bbox import json_to_bboxes  # noqa: E402

_JSON_ONE_BOX = '[{"bbox_2d": [1, 1, 4, 4], "label": "a"}]'


def _frames(batch_sizes, size=16):
    return [torch.zeros((n, size, size, 3), dtype=torch.float32) for n in batch_sizes]


class TestJsonToBBoxesRestructure(unittest.TestCase):
    def setUp(self):
        self.node = json_to_bboxes()

    def _process(self, n_json, batch_sizes):
        return self.node.process([_JSON_ONE_BOX] * n_json, ["simple"], [""], _frames(batch_sizes))

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
        bboxes, image_list = self.node.process([_JSON_ONE_BOX], ["simple"], [""], None)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(image_list, [])


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
        bboxes, _ = self.node.process([_JSON_LABELED], ["simple"], ["cat"], None)
        self.assertEqual(len(bboxes), 1)
        self.assertEqual(bboxes[0], [(1, 1, 4, 4), (3, 3, 6, 6)])

    def test_empty_label_keeps_all(self):
        bboxes, _ = self.node.process([_JSON_LABELED], ["simple"], [""], None)
        self.assertEqual(len(bboxes[0]), 3)

    def test_no_match_returns_empty_group(self):
        bboxes, _ = self.node.process([_JSON_LABELED], ["simple"], ["bird"], None)
        self.assertEqual(bboxes[0], [])

    def test_non_dict_item_with_filter_reports_structure_error(self):
        # 回归: label 过滤开启时非 dict 项(如坐标数组)不得抛裸 AttributeError,
        # 应留给结构校验报出带期望格式的 ValueError
        with self.assertRaises(ValueError):
            self.node.process(['[[10, 20, 30, 40]]'], ["simple"], ["cat"], None)


if __name__ == "__main__":
    unittest.main()
