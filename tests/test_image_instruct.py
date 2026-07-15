"""image Instruct 逐张模式 (_infer_each) 的节点级单元测试.

用 FakeVlm 替身走通 process() 全链路 (不加载真实模型), 锁定:
- 多图输出含 "======== Image N ========" 前缀行, 且经 split_image_results
  可还原为逐张结果列表; 前缀行以字面量断言而非引用模板常量, 拆分正则与
  生成模板同源 (common_static), 同改共错时唯有字面量断言能报出
- 单图输出不加前缀行
- 中断置位时循环立即抛 InterruptProcessingException, 不再发起请求
"""

import types
import unittest
from unittest import mock

import torch

from tests import comfy_stubs

comfy_stubs.install()

from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.i18n.common_static import IMAGE_MODE_EACH  # noqa: E402
from src.nodes.instruct.media.image import node_instruct  # noqa: E402
from src.nodes.instruct.media.image.node_instruct import llama_cpp_image_instruct  # noqa: E402
from src.shared.text_utils import split_image_results  # noqa: E402


class _FakeVlm:
    """process() 全链路所需的最小 llm 替身: 逐次返回预置文本, 附 _run 收尾判定属性."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0
        self.n_tokens = 0
        self._model = types.SimpleNamespace(is_hybrid=lambda: False, is_recurrent=lambda: False)
        self._ctx = mock.Mock()
        self._hybrid_cache_mgr = None

    def abort(self):
        pass

    def create_chat_completion(self, messages, seed, **params):
        text = self._outputs[self.calls]
        self.calls += 1
        return {"choices": [{"message": {"content": text}}]}


class TestImageInstructInferEach(unittest.TestCase):
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

    def _process(self, images):
        (out,) = self.node.process(
            vlm_model=self.config,
            images=images,
            seed=0,
            preset_prompt="空白 - 空",
            custom_prompt="一只猫",
            system_prompt="",
            mode=IMAGE_MODE_EACH,
            max_size=256,
            strip_thinking=True,
            force_offload=False,
        )
        return out

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

    def test_interrupt_raises_before_request(self):
        # 循环起点检查中断标志, 命中即抛且不再发起补全请求
        llm = _FakeVlm(["不应产生"])
        self._install(llm)
        with (
            mock.patch.object(node_instruct.mm, "processing_interrupted", lambda: True),
            self.assertRaises(node_instruct.mm.InterruptProcessingException),
        ):
            self._process(torch.zeros(2, 4, 4, 3))
        self.assertEqual(llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
