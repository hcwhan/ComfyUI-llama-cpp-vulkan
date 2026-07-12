"""app/core/handlers.py 注册表契约的单元测试: 类名存在性, kwargs 合法性, thinking 后缀一致性."""

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

    def test_kwargs_accepted_by_init(self):
        # 全部构造 kwargs 必须被类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError)
        for label, (cls_name, kwargs) in _HANDLER_SPECS.items():
            for key in (kwargs or {}):
                self.assertIn(
                    key, _init_params(cls_name),
                    f'"{label}": {cls_name}.__init__ 不接受 {key}',
                )

    def test_thinking_capable_classes_are_wired(self):
        # 回归: 类签名带 thinking 开关的条目必须在 kwargs 中显式声明开关值,
        # 否则库侧默认值恒生效且用户无法从 UI 控制 (Gemma4/Step3-VL 曾踩此坑)
        for label, (cls_name, kwargs) in _HANDLER_SPECS.items():
            params = _init_params(cls_name)
            capable = [p for p in _THINK_PARAM_NAMES if p in params]
            for param in capable:
                self.assertIn(
                    param, kwargs or {},
                    f'"{label}": {cls_name} 的 __init__ 接受 {param} 但注册表未显式声明',
                )

    def test_thinking_suffix_matches_value(self):
        # -Thinking 后缀 <=> thinking 开关值为 True (类不接受开关的条目豁免,
        # 如 GLM-4.1V-Thinking 的后缀仅描述模型本身)
        for label, (cls_name, kwargs) in _HANDLER_SPECS.items():
            declared = [p for p in _THINK_PARAM_NAMES if p in (kwargs or {})]
            if not declared:
                continue
            expected = label.endswith("-Thinking")
            for param in declared:
                self.assertEqual(
                    kwargs[param], expected,
                    f'"{label}": {param}={kwargs[param]} 与显示名后缀语义不符',
                )

    def test_thinking_variant_shares_class_with_base(self):
        # -Thinking 变体必须与无后缀基名共享同一个类
        for label, (cls_name, _) in _HANDLER_SPECS.items():
            if not label.endswith("-Thinking"):
                continue
            base = label[: -len("-Thinking")]
            if base in _HANDLER_SPECS:
                self.assertEqual(_HANDLER_SPECS[base][0], cls_name)


if __name__ == "__main__":
    unittest.main()
