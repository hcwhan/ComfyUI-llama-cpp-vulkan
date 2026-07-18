"""src/core/handlers.py 注册表契约的单元测试: 类名存在性, kwargs 合法性, thinking 三态元数据与类签名的一致性, 钳制与绑定行为."""

import inspect
import unittest
from unittest import mock

from tests import comfy_stubs

comfy_stubs.install()

import llama_cpp.llama_multimodal as _handler_module  # noqa: E402

from src.core import handlers  # noqa: E402
from src.core.handlers import (  # noqa: E402
    _AUDIO_ONLY_LABELS,
    _HANDLER_SPECS,
    HANDLERS,
    THINK_FORCED,
    THINK_UNSUPPORTED,
    clamp_thinking,
    handler_constructor,
    image_token_handlers,
    thinking_modes,
)
from src.i18n.lang import LANG  # noqa: E402

# 思考相关构造参数名单: 仅收录影响单轮输出的开关
_THINK_PARAM_NAMES = ("enable_thinking", "force_reasoning")

# 多轮对话保留历史思考块类参数的豁免名单 (当前 wheel 有两个:
# LFM25VLChatHandler 的 keep_past_thinking, Qwen35ChatHandler 的
# preserve_thinking) - 本插件无会话状态, 单轮请求不受其影响, 无需接线;
# wheel 新增的其他思考开关名由模糊扫描用例报出
_MULTI_TURN_THINK_EXEMPT = ("keep_past_thinking", "preserve_thinking")

# storage.load_model 构造 handler 时固定注入的 kwargs
_STORAGE_KWARGS = ("mmproj_path", "verbose", "image_max_tokens", "image_min_tokens", "use_gpu")

_SENTINELS = (THINK_UNSUPPORTED, THINK_FORCED)


def _init_params(cls_name):
    return inspect.signature(getattr(_handler_module, cls_name).__init__).parameters


class TestHandlerSpecs(unittest.TestCase):
    def test_all_classes_exist_in_wheel(self):
        for label, (cls_name, _kwargs, _think) in _HANDLER_SPECS.items():
            self.assertTrue(
                hasattr(_handler_module, cls_name),
                f'"{label}": {cls_name} 不在当前 wheel 的 llama_multimodal 中',
            )

    def test_kwargs_accepted_by_init(self):
        # 全部构造 kwargs 必须被类 __init__ 显式接受(基类对未知 kwargs 抛 TypeError)
        for label, (cls_name, kwargs, _think) in _HANDLER_SPECS.items():
            for key in kwargs or {}:
                self.assertIn(
                    key,
                    _init_params(cls_name),
                    f'"{label}": {cls_name}.__init__ 不接受 {key}',
                )

    def test_toggle_param_accepted_by_init(self):
        # 可切换档声明的参数名必须被类 __init__ 显式接受
        for label, (cls_name, _kwargs, think) in _HANDLER_SPECS.items():
            if think in _SENTINELS:
                continue
            self.assertIn(
                think,
                _init_params(cls_name),
                f'"{label}": {cls_name}.__init__ 不接受三态元数据声明的 {think}',
            )

    def test_non_toggle_classes_accept_no_think_param(self):
        # 不支持/强制档的类签名不得含思考开关参数, 否则该条目应改为可切换档
        # (防 wheel 升级给类加开关而注册表未接线, 库侧默认值恒生效且 UI 不可控)
        for label, (cls_name, _kwargs, think) in _HANDLER_SPECS.items():
            if think not in _SENTINELS:
                continue
            params = _init_params(cls_name)
            for param in _THINK_PARAM_NAMES:
                self.assertNotIn(
                    param,
                    params,
                    f'"{label}": {cls_name}.__init__ 接受 {param} 但三态元数据标注为 {think}',
                )

    def test_fuzzy_scan_flags_unaccounted_think_params(self):
        # 防 wheel 新增第三种思考开关名 (_THINK_PARAM_NAMES 是静态清单):
        # 扫描全部注册类签名, 参数名含 think/reason 且既非该条目声明的
        # 可切换参数也不在多轮豁免名单内的, 一律报出提示复核
        # (确认是单轮开关则改注册表三态接线, 是多轮参数则入豁免名单)
        for label, (cls_name, _kwargs, think) in _HANDLER_SPECS.items():
            declared = think if think not in _SENTINELS else None
            for param in _init_params(cls_name):
                if "think" not in param and "reason" not in param:
                    continue
                self.assertTrue(
                    param == declared or param in _MULTI_TURN_THINK_EXEMPT,
                    f'"{label}": {cls_name}.__init__ 的 {param} 是未接线的思考开关, 复核应改为可切换档还是加入多轮豁免名单',
                )

    def test_thinking_suffix_only_on_forced(self):
        # "-Thinking" 后缀只允许出现在强制思考档 (后缀描述模型本身,
        # 如 GLM-4.1V-Thinking); 可切换档由 VLM Loader 的 thinking 开关承载
        for label, (_cls_name, _kwargs, think) in _HANDLER_SPECS.items():
            if label.endswith("-Thinking"):
                self.assertEqual(think, THINK_FORCED, f'"{label}": 带 -Thinking 后缀但非强制思考档')

    def test_storage_construction_kwargs_reach_every_class(self):
        # storage.load_model 对所有 handler 注入五个构造 kwargs, 可用性依赖
        # 子类签名显式接受或经 **kwargs 透传到基类; 基类必须显式接受兜底,
        # wheel 升级时某子类改签名不再透传在此报出而非等到运行期
        base_params = inspect.signature(_handler_module.MTMDChatHandler.__init__).parameters
        for key in _STORAGE_KWARGS:
            self.assertIn(key, base_params, f"基类 MTMDChatHandler.__init__ 不接受 {key}")
        for label, (cls_name, _kwargs, _think) in _HANDLER_SPECS.items():
            params = _init_params(cls_name)
            has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            for key in _STORAGE_KWARGS:
                self.assertTrue(
                    key in params or has_var_kw,
                    f'"{label}": {cls_name}.__init__ 既不显式接受 {key} 也无 **kwargs 透传',
                )


class TestClampThinking(unittest.TestCase):
    def test_unsupported_true_clamped_off_with_warning(self):
        with mock.patch.object(handlers.logger, "warning") as warn:
            self.assertFalse(clamp_thinking("Gemma3", True))
        warn.assert_called_once()

    def test_forced_false_clamped_on_with_warning(self):
        with mock.patch.object(handlers.logger, "warning") as warn:
            self.assertTrue(clamp_thinking("GLM-4.1V-Thinking", False))
        warn.assert_called_once()

    def test_legal_values_pass_through_silently(self):
        with mock.patch.object(handlers.logger, "warning") as warn:
            self.assertFalse(clamp_thinking("Gemma3", False))
            self.assertTrue(clamp_thinking("GLM-4.1V-Thinking", True))
            self.assertTrue(clamp_thinking("Qwen3.6", True))
            self.assertFalse(clamp_thinking("Qwen3.6", False))
        warn.assert_not_called()


class TestHandlerConstructor(unittest.TestCase):
    def test_toggle_binds_declared_param(self):
        ctor = handler_constructor("Qwen3.6", True)
        self.assertEqual(ctor.keywords, {"enable_thinking": True})
        ctor = handler_constructor("Qwen3-VL", False)
        self.assertEqual(ctor.keywords, {"force_reasoning": False})

    def test_non_toggle_returns_base_constructor(self):
        # 不支持/强制档的类没有开关参数, 原样返回注册表构造器
        self.assertIs(handler_constructor("Gemma3", True), HANDLERS["Gemma3"])
        self.assertIs(handler_constructor("GLM-4.1V-Thinking", False), HANDLERS["GLM-4.1V-Thinking"])

    def test_unknown_label_raises_key_error(self):
        with self.assertRaises(KeyError):
            handler_constructor("No-Such-Handler", False)


class TestThinkingModes(unittest.TestCase):
    def test_keys_match_available_handlers(self):
        self.assertEqual(set(thinking_modes()), set(HANDLERS))

    def test_values_are_tri_state(self):
        modes = thinking_modes()
        self.assertLessEqual(set(modes.values()), {"toggle", "forced", "none"})
        self.assertEqual(modes["Qwen3.6"], "toggle")
        self.assertEqual(modes["GLM-4.1V-Thinking"], "forced")
        self.assertEqual(modes["Gemma3"], "none")

    def test_missing_handler_excluded(self):
        # "只含可用 handler" 语义: 从 HANDLERS 摘除一个条目模拟 wheel 缺类
        # (_HANDLER_SPECS 仍含它), 名单须跟随消失; 全类可用环境下
        # test_keys_match_available_handlers 无法区分实现遍历的是哪张表,
        # 若实现改为遍历 _HANDLER_SPECS 则在此报出
        pruned = {label: ctor for label, ctor in HANDLERS.items() if label != "Qwen3.6"}
        with mock.patch.object(handlers, "HANDLERS", pruned):
            self.assertNotIn("Qwen3.6", thinking_modes())


class TestImageTokenHandlers(unittest.TestCase):
    def test_audio_only_labels_exist_in_specs(self):
        # 防注册表改名后名单指向不存在的条目而静默失效
        self.assertLessEqual(_AUDIO_ONLY_LABELS, set(_HANDLER_SPECS))

    def test_audio_only_excluded_others_kept(self):
        labels = image_token_handlers()
        self.assertNotIn("(ASR) Qwen3-ASR", labels)
        self.assertIn("Qwen3-VL", labels)
        self.assertIn("-Generic-", labels)
        self.assertEqual(set(labels) | _AUDIO_ONLY_LABELS, set(HANDLERS))

    def test_missing_handler_excluded(self):
        # 同 TestThinkingModes.test_missing_handler_excluded: 名单只含
        # 可用 handler, 缺类条目 (仍在 _HANDLER_SPECS 中) 不得出现
        pruned = {label: ctor for label, ctor in HANDLERS.items() if label != "Qwen3-VL"}
        with mock.patch.object(handlers, "HANDLERS", pruned):
            self.assertNotIn("Qwen3-VL", image_token_handlers())


class TestResolveHandlers(unittest.TestCase):
    def test_missing_class_dropped_others_kept(self):
        # wheel 升级导致类缺失时: 该选项从下拉框剔除 (打 warning),
        # 不静默吞错也不阻断整个注册表; assertLogs 兼收敛测试输出
        specs = {
            "Fake-Handler": ("NoSuchHandlerClass", None, handlers.THINK_UNSUPPORTED),
            "Gemma3": ("Gemma3ChatHandler", None, handlers.THINK_UNSUPPORTED),
        }
        with mock.patch.object(handlers, "_HANDLER_SPECS", specs), self.assertLogs("llama-cpp-vulkan", level="WARNING") as logs:
            resolved = handlers._resolve_handlers()
        self.assertNotIn("Fake-Handler", resolved)
        self.assertIn("Gemma3", resolved)
        expected = LANG["logs"]["handlers"]["handlers_unavailable"].format(missing="Fake-Handler (NoSuchHandlerClass)")
        self.assertTrue(any(expected in m for m in logs.output))

    def test_kwargs_prebound_via_partial(self):
        # 声明了固定 kwargs 的条目须经 functools.partial 预绑定构造参数;
        # .func 同一性锁定绑定的确是注册表声明的那个类
        specs = {"-Generic-": ("GenericMTMDChatHandler", {"chat_format": None}, handlers.THINK_UNSUPPORTED)}
        with mock.patch.object(handlers, "_HANDLER_SPECS", specs):
            resolved = handlers._resolve_handlers()
        self.assertIs(resolved["-Generic-"].func, _handler_module.GenericMTMDChatHandler)
        self.assertEqual(resolved["-Generic-"].keywords, {"chat_format": None})


if __name__ == "__main__":
    unittest.main()
