"""src/core/handlers.py 注册表契约的单元测试: 类名存在性, kwargs 合法性, thinking 后缀一致性."""

import inspect
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

import llama_cpp.llama_multimodal as _handler_module  # noqa: E402

from src.core import handlers  # noqa: E402
from src.core.handlers import _HANDLER_SPECS  # noqa: E402

# 思考相关构造参数名单: 仅收录影响单轮输出的开关。多轮对话保留历史思考块
# 类参数豁免不收录 (当前 wheel 有两个: LFM25VLChatHandler 的 keep_past_thinking、
# Qwen3VLChatHandler 的 preserve_thinking) —— 本插件无会话状态, 单轮请求不受
# 其影响, 无需接线; 升级 wheel 复核时顺带扫一遍新 handler 签名有无新增思考开关
_THINK_PARAM_NAMES = ("enable_thinking", "force_reasoning")

# storage.load_model 构造 handler 时固定注入的 kwargs
_STORAGE_KWARGS = ("mmproj_path", "verbose", "image_max_tokens", "image_min_tokens", "use_gpu")


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
            for key in kwargs or {}:
                self.assertIn(
                    key,
                    _init_params(cls_name),
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
                    param,
                    kwargs or {},
                    f'"{label}": {cls_name} 的 __init__ 接受 {param} 但注册表未显式声明',
                )

    def test_thinking_suffix_matches_value(self):
        # -Thinking 后缀 <=> thinking 开关值为 True (类不接受开关的条目豁免,
        # 如 GLM-4.1V-Thinking 的后缀仅描述模型本身)
        for label, (_cls_name, kwargs) in _HANDLER_SPECS.items():
            declared = [p for p in _THINK_PARAM_NAMES if p in (kwargs or {})]
            if not declared:
                continue
            expected = label.endswith("-Thinking")
            for param in declared:
                self.assertEqual(
                    kwargs[param],
                    expected,
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

    def test_storage_construction_kwargs_reach_every_class(self):
        # storage.load_model 对所有 handler 注入五个构造 kwargs, 可用性依赖
        # 子类签名显式接受或经 **kwargs 透传到基类; 基类必须显式接受兜底,
        # wheel 升级时某子类改签名不再透传在此报出而非等到运行期
        base_params = inspect.signature(_handler_module.MTMDChatHandler.__init__).parameters
        for key in _STORAGE_KWARGS:
            self.assertIn(key, base_params, f"基类 MTMDChatHandler.__init__ 不接受 {key}")
        for label, (cls_name, _) in _HANDLER_SPECS.items():
            params = _init_params(cls_name)
            has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            for key in _STORAGE_KWARGS:
                self.assertTrue(
                    key in params or has_var_kw,
                    f'"{label}": {cls_name}.__init__ 既不显式接受 {key} 也无 **kwargs 透传',
                )


class TestResolveHandlers(unittest.TestCase):
    def test_missing_class_dropped_others_kept(self):
        # wheel 升级导致类缺失时: 该选项从下拉框剔除 (打 warning),
        # 不静默吞错也不阻断整个注册表
        specs = {
            "Fake-Handler": ("NoSuchHandlerClass", None),
            "Gemma3": ("Gemma3ChatHandler", None),
        }
        with mock.patch.object(handlers, "_HANDLER_SPECS", specs):
            resolved = handlers._resolve_handlers()
        self.assertNotIn("Fake-Handler", resolved)
        self.assertIn("Gemma3", resolved)

    def test_kwargs_prebound_via_partial(self):
        # 声明了 kwargs 的条目须经 functools.partial 预绑定构造参数
        specs = {"Gemma4": ("Gemma4ChatHandler", {"enable_thinking": False})}
        with mock.patch.object(handlers, "_HANDLER_SPECS", specs):
            resolved = handlers._resolve_handlers()
        self.assertEqual(resolved["Gemma4"].keywords, {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
