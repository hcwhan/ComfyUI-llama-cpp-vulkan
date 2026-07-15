"""src/nodes/bbox/ bboxes_to_segs / bboxes_to_mask 的节点级单元测试.

覆盖: 坐标裁剪, dilation/crop_factor 几何, SEG 字段组装 (Impact Pack 兼容),
无效框逐个跳过, 零面积框 + dilation 两节点行为一致, 多帧取首帧,
mask 的羽化衰减与多框 maximum 合成, segs 羽化与 mask 节点衰减一致, 恒 CPU 输出.
"""

import unittest

import numpy as np
import torch

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.bbox.node_bboxes_to_mask import bboxes_to_mask  # noqa: E402
from src.nodes.bbox.node_bboxes_to_segs import bboxes_to_segs  # noqa: E402


class TestBBoxesToSegs(unittest.TestCase):
    def _process(self, bboxes, image=None, label="person", confidence=0.8, dilation=0, feather=0, crop_factor=1.0):
        image = image if image is not None else torch.zeros(1, 32, 32, 3)
        return bboxes_to_segs().process(bboxes, image, label, confidence, dilation, feather, crop_factor)[0]

    def test_seg_fields_and_shapes(self):
        shape, seg_list = self._process([(4, 4, 12, 12)])
        self.assertEqual(shape, (32, 32))
        self.assertEqual(len(seg_list), 1)
        seg = seg_list[0]
        np.testing.assert_array_equal(seg.bbox, np.array([4, 4, 12, 12], dtype=np.float32))
        self.assertEqual(seg.bbox.dtype, np.float32)
        self.assertEqual(seg.label, "person")
        self.assertEqual(seg.confidence, 0.8)
        self.assertEqual(seg.crop_region, [4, 4, 12, 12])
        self.assertEqual(tuple(seg.cropped_image.shape), (1, 8, 8, 3))
        # 恒 CPU 输出 (与 bboxes_to_mask / json_to_bboxes 策略一致)
        self.assertEqual(seg.cropped_image.device.type, "cpu")
        self.assertEqual(seg.cropped_mask.shape, (8, 8))
        self.assertTrue((seg.cropped_mask == 1.0).all())
        self.assertIsNone(seg.control_net_wrapper)

    def test_dilation_expands_mask_and_crop(self):
        _, seg_list = self._process([(4, 4, 12, 12)], dilation=2)
        seg = seg_list[0]
        # 掩码矩形外扩到 (2,2,14,14), crop_factor=1.0 时 crop_region 与其一致
        self.assertEqual(seg.crop_region, [2, 2, 14, 14])
        self.assertEqual(seg.cropped_mask.shape, (12, 12))
        self.assertTrue((seg.cropped_mask == 1.0).all())
        # dilation 不改变 seg.bbox (Impact Pack 约定保留原始检测框)
        np.testing.assert_array_equal(seg.bbox, np.array([4, 4, 12, 12], dtype=np.float32))

    def test_crop_factor_adds_context_window(self):
        _, seg_list = self._process([(4, 4, 12, 12)], crop_factor=3.0)
        seg = seg_list[0]
        # 8x8 掩码矩形按 3 倍外扩: pad 8, 裁剪到图像内
        self.assertEqual(seg.crop_region, [0, 0, 20, 20])
        self.assertEqual(seg.cropped_mask.shape, (20, 20))
        # 掩码矩形在 crop 窗口内的相对位置仍是 (4,4,12,12)
        self.assertEqual(seg.cropped_mask[8, 8], 1.0)
        self.assertEqual(seg.cropped_mask[0, 0], 0.0)

    def test_out_of_bounds_box_clamped(self):
        _, seg_list = self._process([(-5, -5, 10, 10)])
        seg = seg_list[0]
        np.testing.assert_array_equal(seg.bbox, np.array([0, 0, 10, 10], dtype=np.float32))
        self.assertEqual(tuple(seg.cropped_image.shape), (1, 10, 10, 3))

    def test_invalid_and_degenerate_boxes_skipped(self):
        shape, seg_list = self._process([(1, 2, 3), "junk", (5, 5, 5, 5), (40, 40, 50, 50)])
        self.assertEqual(shape, (32, 32))
        self.assertEqual(seg_list, [])

    def test_zero_area_bbox_dilated_consistent_with_mask(self):
        # 回归: 零面积框 + dilation 时先外扩后判空, 与 bboxes_to_mask 行为一致,
        # 外扩出的区域不被提前丢弃
        _, seg_list = self._process([(5, 5, 5, 5)], dilation=10)
        self.assertEqual(len(seg_list), 1)
        seg = seg_list[0]
        # 扩张框 (-5,-5,15,15) 裁剪到图像内 -> 掩码矩形 (0,0,15,15)
        self.assertEqual(seg.crop_region, [0, 0, 15, 15])
        self.assertEqual(seg.cropped_mask.shape, (15, 15))
        self.assertTrue((seg.cropped_mask == 1.0).all())
        # seg.bbox 仍保留原始检测框, 不含 dilation
        np.testing.assert_array_equal(seg.bbox, np.array([5, 5, 5, 5], dtype=np.float32))

        # 同一输入在 MASK 节点画出同一块区域
        mask = bboxes_to_mask().process([(5, 5, 5, 5)], torch.zeros(1, 32, 32, 3), 10, 0)[0]
        self.assertEqual(mask[0, :15, :15].min().item(), 1.0)
        self.assertEqual(mask.sum().item(), 15 * 15)

    def test_feather_decay_matches_mask_node(self):
        # 回归: 羽化在扩了 margin 的窗口上滤波后裁回 crop_region 尺寸,
        # 贴近 crop 窗口边缘的衰减不再被 reflect 边界抬高;
        # 同参数下衰减形状与 bboxes_to_mask 一致
        bboxes = [(8, 8, 24, 24)]
        _, seg_list = self._process(bboxes, dilation=0, feather=2, crop_factor=1.0)
        seg = seg_list[0]
        self.assertEqual(seg.crop_region, [8, 8, 24, 24])
        # cropped_mask 形状仍与 crop_region 一致 (Impact Pack 约定)
        self.assertEqual(seg.cropped_mask.shape, (16, 16))
        # 旧实现: 掩码矩形填满 crop 窗口, reflect 反射使整窗恒为 1.0
        self.assertGreater(seg.cropped_mask[8, 8], 0.9)
        self.assertLess(seg.cropped_mask[0, 8], 1.0)

        mask = bboxes_to_mask().process(bboxes, torch.zeros(1, 32, 32, 3), 0, 2)[0]
        cx1, cy1, cx2, cy2 = seg.crop_region
        np.testing.assert_allclose(seg.cropped_mask, mask[0, cy1:cy2, cx1:cx2].numpy(), atol=1e-6)

    def test_batch_crops_from_first_frame(self):
        image = torch.cat([torch.zeros(1, 32, 32, 3), torch.ones(1, 32, 32, 3)])
        _, seg_list = self._process([(4, 4, 12, 12)], image=image)
        self.assertEqual(seg_list[0].cropped_image.sum().item(), 0.0)


class TestBBoxesToMask(unittest.TestCase):
    def _process(self, bboxes, dilation=0, feather=0, h=32, w=32):
        image = torch.zeros(1, h, w, 3)
        return bboxes_to_mask().process(bboxes, image, dilation, feather)[0]

    def test_rect_mask_basic(self):
        mask = self._process([(4, 4, 12, 12)])
        self.assertEqual(tuple(mask.shape), (1, 32, 32))
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(mask.device.type, "cpu")
        self.assertEqual(mask[0, 8, 8].item(), 1.0)
        self.assertEqual(mask[0, 2, 2].item(), 0.0)

    def test_dilation_expands_rect(self):
        mask = self._process([(4, 4, 12, 12)], dilation=2)
        self.assertEqual(mask[0, 3, 3].item(), 1.0)
        self.assertEqual(mask[0, 1, 1].item(), 0.0)

    def test_feather_smooths_monotonically(self):
        mask = self._process([(8, 8, 24, 24)], feather=2)
        center = mask[0, 16, 16].item()
        edge = mask[0, 8, 16].item()
        corner = mask[0, 0, 0].item()
        self.assertGreater(center, edge)
        self.assertGreater(edge, corner)
        self.assertTrue(0.0 < edge < 1.0)
        self.assertLess(corner, 0.1)
        self.assertLessEqual(mask.max().item(), 1.0)

    def test_invalid_and_outside_boxes_skipped(self):
        mask = self._process([(1, 2, 3), (40, 40, 50, 50)])
        self.assertEqual(mask.sum().item(), 0.0)

    def test_overlapping_boxes_max_combined(self):
        mask = self._process([(4, 4, 12, 12), (8, 8, 16, 16)])
        # maximum 合成: 重叠区不叠加, 总量 = 并集面积
        self.assertEqual(mask.max().item(), 1.0)
        self.assertEqual(mask.sum().item(), 8 * 8 + 8 * 8 - 4 * 4)


if __name__ == "__main__":
    unittest.main()
