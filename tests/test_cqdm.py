"""src/core/cqdm.py 的单元测试: 构造即推 0/N, 迭代推进次序与提前退出收尾."""

import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

import comfy.utils  # noqa: E402

from src.core.cqdm import cqdm  # noqa: E402


class _RecordingBar:
    """记录 update_absolute 调用的 ProgressBar 替身 (stub 版分不出是否推送过)."""

    def __init__(self, total, node_id=None):
        self.total = total
        self.current = 0
        self.absolute_calls = []

    def update(self, value):
        self.current += value

    def update_absolute(self, value, total=None, preview=None):
        self.absolute_calls.append(value)
        self.current = value


class TestCqdm(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(comfy.utils, "ProgressBar", _RecordingBar)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_constructor_pushes_zero(self):
        # 回归 (b73dc62): 构造即推 0/N, 清掉上次执行被中断时残留的前端进度
        bar = cqdm([1, 2, 3])
        self.assertEqual(bar.pbar.absolute_calls, [0])
        self.assertEqual(bar.pbar.total, 3)
        self.assertEqual(len(bar), 3)
        list(bar)

    def test_progress_advances_after_each_item(self):
        # 第 N 项处理完才前进到 N: 处理某项期间进度停留在上一项的计数
        bar = cqdm(["a", "b"])
        seen = []
        for item in bar:
            seen.append((item, bar.pbar.current))
        self.assertEqual(seen, [("a", 0), ("b", 1)])
        self.assertEqual(bar.pbar.current, 2)

    def test_tqdm_closed_on_early_exit(self):
        # 中断/异常提前退出时 finally 收尾 tqdm (close 后 disable 置 True),
        # 终端不残留未完成的进度条
        bar = cqdm([1, 2, 3])
        it = iter(bar)
        next(it)
        it.close()
        self.assertTrue(bar.tqdm.disable)
        self.assertEqual(bar.pbar.current, 0)


if __name__ == "__main__":
    unittest.main()
