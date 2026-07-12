"""app/core/handlers.py 注册表契约的单元测试: 类名存在性与 thinking 开关接管的一致性."""

import inspect
import unittest

from tests import comfy_stubs

comfy_stubs.install()

import llama_cpp.llama_multimodal as _handler_module  # noqa: E402

from app.core.handlers import _HANDLER_SPECS  # noqa: E402

_THINK_PARAM_NAMES = ("enable_thinking", "force_reasoning")


def _init_params(cls_name):
    return inspect.signature(getattr(_handler_module, cls_name).__init__).parameters


class TestHandlerSpecs(unittest.TestCase):
    def test_all_classes_exist_in_wheel(self):
        for label, (cls_name, _) in _HANDLER_SPECS.items():
            self.assertTrue(
                hasattr(_handler_module, cls_name),
                f'"{label}": {cls_name} 不在当前 wheel 的 llama_multimodal 中',
            )

    def test_think_param_accepted_by_init(self):
        # think_param 必须被类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError)
        for label, (cls_name, think_param) in _HANDLER_SPECS.items():
            if think_param is None:
                continue
            self.assertIn(
                think_param, _init_params(cls_name),
                f'"{label}": {cls_name}.__init__ 不接受 {think_param}',
            )

    def test_thinking_capable_classes_are_wired(self):
        # 回归: 类签名带 thinking 开关的条目不得填 None,
        # 否则库侧默认值恒生效且用户无法从 UI 关闭 (Gemma4/Step3-VL 曾踩此坑)
        for label, (cls_name, think_param) in _HANDLER_SPECS.items():
            params = _init_params(cls_name)
            capable = [p for p in _THINK_PARAM_NAMES if p in params]
            if capable:
                self.assertIsNotNone(
                    think_param,
                    f'"{label}": {cls_name} 的 __init__ 接受 {capable} 但注册表未接管',
                )

    def test_thinking_suffix_pairs_share_class(self):
        # -Thinking 变体必须与无后缀条目共享同一个类与参数名
        for label, (cls_name, think_param) in _HANDLER_SPECS.items():
            if not label.endswith("-Thinking"):
                continue
            base = label[: -len("-Thinking")]
            if base in _HANDLER_SPECS:
                self.assertEqual(_HANDLER_SPECS[base], (cls_name, think_param))


if __name__ == "__main__":
    unittest.main()
