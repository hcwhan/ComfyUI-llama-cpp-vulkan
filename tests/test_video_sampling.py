"""video Instruct 抽帧逻辑的单元测试: linspace 均匀采样与 clamp 边界."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from app.nodes.type.media.video.node_instruct import sample_frame_indices  # noqa: E402


class TestSampleFrameIndices(unittest.TestCase):
    def test_uniform_sampling(self):
        self.assertEqual(list(sample_frame_indices(10, 4)), [0, 3, 6, 9])

    def test_clamp_to_total_no_duplicates(self):
        # 回归: max_frames 超过实际帧数时 clamp, 不得重复采样同一帧
        self.assertEqual(list(sample_frame_indices(3, 24)), [0, 1, 2])

    def test_single_frame(self):
        self.assertEqual(list(sample_frame_indices(1, 24)), [0])

    def test_first_and_last_frame_included(self):
        indices = list(sample_frame_indices(100, 5))
        self.assertEqual(len(indices), 5)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)


if __name__ == "__main__":
    unittest.main()
