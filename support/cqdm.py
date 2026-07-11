import sys

import comfy.utils
from tqdm import tqdm


class cqdm:
    """迭代器封装：同时驱动 ComfyUI 前端的 ProgressBar 和终端的 tqdm。"""

    def __init__(self, iterable, desc="Processing"):
        self.total = len(iterable)
        self.pbar = comfy.utils.ProgressBar(self.total)
        self.tqdm = tqdm(
            iterable=iterable,
            total=self.total,
            desc=desc,
            dynamic_ncols=True,
            file=sys.stdout,
        )

    def __iter__(self):
        for item in self.tqdm:
            self.pbar.update(1)
            yield item

    def __len__(self):
        return self.total
