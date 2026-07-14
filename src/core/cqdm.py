"""进度条封装, 一个迭代器同时驱动 ComfyUI 前端 ProgressBar 和终端 tqdm."""

import comfy.utils
from tqdm import tqdm


class cqdm:
    def __init__(self, iterable, desc="Processing"):
        self.total = len(iterable)
        self.pbar = comfy.utils.ProgressBar(self.total)
        # 立即推送 0/N: 前端马上显示进度起点, 同时清掉上一次执行被中断时
        # 残留的进度值. 中断时刻无法归零 - ComfyUI 的进度 hook 在发送前
        # 检查中断标志并直接抛 InterruptProcessingException, 推送必然失败,
        # 因此归零只能推迟到下一次执行的起点
        self.pbar.update_absolute(0)
        self.tqdm = tqdm(
            iterable=iterable,
            total=self.total,
            desc=desc,
            dynamic_ncols=True,
        )

    def __iter__(self):
        # 中断/异常时确保 tqdm 收尾, 避免终端残留未完成的进度条
        try:
            for item in self.tqdm:
                yield item
                # 放在 yield 之后: 第 N 项处理完才前进到 N, 与 tqdm 的计数语义一致
                self.pbar.update(1)
        finally:
            self.tqdm.close()

    def __len__(self):
        return self.total
