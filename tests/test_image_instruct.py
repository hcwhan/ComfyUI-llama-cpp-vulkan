"""image Instruct 逐张 (_infer_each) 与批量 (_infer_batch) 模式的节点级单元测试.

用 FakeVlm 替身走通 process() 全链路 (不加载真实模型), 锁定:
- 逐张模式: 多图输出含 "======== Image N ========" 前缀行, 且经
  split_image_results 可还原为逐张结果列表; 前缀行以字面量断言而非引用
  模板常量, 拆分正则与生成模板同源 (common_static), 同改共错时唯有
  字面量断言能报出
- 逐张模式: 单图输出不加前缀行
- 逐张模式: 中断置位时循环立即抛 InterruptProcessingException, 不再发起请求
- 逐张模式: increment_seed 关闭 (默认) 时各请求复用同一 seed, 开启时第 i 张
  用 seed+i 派生, 且回绕避开 llama.cpp 的随机种子哨兵值 0xFFFFFFFF
- 逐张模式: 每次请求各产生一条生成统计日志 (经 _completion_with_stats)
- 批量模式: 全部图片并入单条 user 消息 (文本项 + N 个 image_url 项),
  多图逐张缩放到 max_size, 单图保持原分辨率 (tooltip 承诺 "仅在发送
  多张图片时生效")
"""

import base64
import io
import itertools
import types
import unittest
from unittest import mock

import torch
from PIL import Image

from tests import comfy_stubs

comfy_stubs.install()

from src.core import instruct as core_instruct  # noqa: E402
from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.i18n.common_static import IMAGE_MODE_BATCH, IMAGE_MODE_EACH  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402
from src.nodes.instruct.media.image import node_instruct  # noqa: E402
from src.nodes.instruct.media.image.node_instruct import llama_cpp_image_instruct  # noqa: E402
from src.shared.logger import node_log_prefix  # noqa: E402
from src.shared.text_utils import split_image_results  # noqa: E402


class _FakeVlm:
    """process() 全链路所需的最小 llm 替身: 逐次返回预置文本, 附 _run 收尾判定属性."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0
        self.seeds = []
        self.last_messages = None
        self.n_tokens = 0
        self._model = types.SimpleNamespace(is_hybrid=lambda: False, is_recurrent=lambda: False)
        self._ctx = mock.Mock()
        self._hybrid_cache_mgr = None

    def abort(self):
        pass

    def create_chat_completion(self, messages, seed, **params):
        self.last_messages = messages
        self.seeds.append(seed)
        text = self._outputs[self.calls]
        self.calls += 1
        return {
            "usage": {"prompt_tokens": 20, "completion_tokens": 100},
            "choices": [{"message": {"content": text}}],
        }


def _png_size(content_item):
    """image_url 内容项的 data URL -> PNG 实际 (宽, 高), 用于断言缩放行为."""
    b64 = content_item["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


class _ImageInstructTestBase(unittest.TestCase):
    MODE = IMAGE_MODE_EACH

    def setUp(self):
        self.node = llama_cpp_image_instruct()
        self.config = {"model": "m.gguf"}
        self._orig_state = (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config)
        self.addCleanup(self._restore_state)

    def _restore_state(self):
        (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config) = self._orig_state

    def _install(self, llm):
        # current_config 与 llama_model 相同, _prepare_messages 不触发真实加载;
        # chat_handler 带 mmproj_path 满足 require_mmproj 校验
        LLAMA_CPP_STORAGE.llm = llm
        LLAMA_CPP_STORAGE.chat_handler = types.SimpleNamespace(mmproj_path="fake.mmproj")
        LLAMA_CPP_STORAGE.current_config = self.config

    def _process(self, images, max_size=256, seed=0, increment_seed=False):
        (out,) = self.node.process(
            vlm_model=self.config,
            images=images,
            seed=seed,
            preset_prompt="空白 - 空",
            custom_prompt="一只猫",
            system_prompt="",
            mode=self.MODE,
            increment_seed=increment_seed,
            max_size=max_size,
            strip_thinking=True,
            force_offload=False,
        )
        return out


class TestImageInstructInferEach(_ImageInstructTestBase):
    def test_multi_image_prefixed_and_splittable(self):
        llm = _FakeVlm(["第一张结果", "第二张结果"])
        self._install(llm)
        out = self._process(torch.zeros(2, 4, 4, 3))
        self.assertEqual(llm.calls, 2)
        self.assertIn("======== Image 1 ========", out)
        self.assertIn("======== Image 2 ========", out)
        self.assertEqual(split_image_results(out), ["第一张结果", "第二张结果"])

    def test_single_image_no_prefix_line(self):
        llm = _FakeVlm(["唯一结果"])
        self._install(llm)
        out = self._process(torch.zeros(1, 4, 4, 3))
        self.assertEqual(llm.calls, 1)
        self.assertEqual(out, "唯一结果")
        self.assertEqual(split_image_results(out), ["唯一结果"])

    def test_same_seed_reused_by_default(self):
        llm = _FakeVlm(["a", "b"])
        self._install(llm)
        self._process(torch.zeros(2, 4, 4, 3), seed=7)
        self.assertEqual(llm.seeds, [7, 7])

    def test_increment_seed_derives_per_image(self):
        llm = _FakeVlm(["a", "b", "c"])
        self._install(llm)
        self._process(torch.zeros(3, 4, 4, 3), seed=7, increment_seed=True)
        self.assertEqual(llm.seeds, [7, 8, 9])

    def test_increment_seed_wraps_around_sentinel(self):
        # seed 上限 0xFFFFFFFE, +1 后回绕到 0, 不落在哨兵值 0xFFFFFFFF
        llm = _FakeVlm(["a", "b"])
        self._install(llm)
        self._process(torch.zeros(2, 4, 4, 3), seed=0xFFFFFFFE, increment_seed=True)
        self.assertEqual(llm.seeds, [0xFFFFFFFE, 0])

    def test_each_mode_logs_stats_per_image(self):
        # 逐张模式每次请求各产生一条生成统计日志 (经 _completion_with_stats);
        # perf_counter 打桩为每调一次 +1 秒, 单次 completion 首尾相邻两次调用差
        # 恒为 1 秒: completion 100 tokens -> 速度 100.0 tok/s
        llm = _FakeVlm(["第一张结果", "第二张结果"])
        self._install(llm)
        ticks = itertools.count()
        expected = node_log_prefix(llama_cpp_image_instruct.LOG_NAME) + LANG["logs"]["instruct"]["generation_stats"].format(
            prompt_tokens=20, completion_tokens=100, elapsed=1.0, speed=100.0
        )
        with (
            mock.patch.object(core_instruct.time, "perf_counter", side_effect=lambda: float(next(ticks))),
            self.assertLogs("llama-cpp-vulkan", level="INFO") as captured,
        ):
            self._process(torch.zeros(2, 4, 4, 3))
        stats_messages = [r.getMessage() for r in captured.records if r.getMessage() == expected]
        self.assertEqual(len(stats_messages), 2)

    def test_interrupt_raises_before_request(self):
        # 循环起点检查中断标志, 命中即抛且不再发起 completion 请求
        llm = _FakeVlm(["不应产生"])
        self._install(llm)
        with (
            mock.patch.object(node_instruct.mm, "processing_interrupted", lambda: True),
            self.assertRaises(node_instruct.mm.InterruptProcessingException),
        ):
            self._process(torch.zeros(2, 4, 4, 3))
        self.assertEqual(llm.calls, 0)


class TestImageInstructInferBatch(_ImageInstructTestBase):
    MODE = IMAGE_MODE_BATCH

    def _user_content(self, llm):
        (user_msg,) = [m for m in llm.last_messages if m["role"] == "user"]
        return user_msg["content"]

    def test_multi_image_merged_into_single_message_and_scaled(self):
        # 多图并入单条 user 消息 (文本项 + N 个 image_url 项), 一次推理;
        # 两张 H8xW16 图按 max_size=8 逐张缩放为 H4xW8
        # (断言取 PIL (宽, 高) 即 (8, 4))
        llm = _FakeVlm(["批量结果"])
        self._install(llm)
        out = self._process(torch.zeros(2, 8, 16, 3), max_size=8)
        self.assertEqual(out, "批量结果")
        self.assertEqual(llm.calls, 1)
        content = self._user_content(llm)
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url"])
        self.assertEqual(content[0]["text"], "一只猫")
        for item in content[1:]:
            self.assertEqual(_png_size(item), (8, 4))

    def test_single_image_keeps_original_resolution(self):
        # 单图保持原分辨率: 16 宽超出 max_size=8 也不缩放
        # (max_size tooltip 承诺 "仅在发送多张图片时生效")
        llm = _FakeVlm(["单图结果"])
        self._install(llm)
        self._process(torch.zeros(1, 8, 16, 3), max_size=8)
        self.assertEqual(llm.calls, 1)
        content = self._user_content(llm)
        self.assertEqual([item["type"] for item in content], ["text", "image_url"])
        self.assertEqual(_png_size(content[1]), (16, 8))

    def test_small_images_not_upscaled(self):
        # 不超 max_size 的多图不做等尺寸重采样, 原分辨率进消息
        llm = _FakeVlm(["批量结果"])
        self._install(llm)
        self._process(torch.zeros(2, 4, 4, 3), max_size=8)
        content = self._user_content(llm)
        for item in content[1:]:
            self.assertEqual(_png_size(item), (4, 4))


if __name__ == "__main__":
    unittest.main()
