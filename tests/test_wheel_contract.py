"""wheel 私有 API 契约测试: 静态检查插件依赖的 llama-cpp-python 内部接口存在性.

不加载模型, 仅做属性/签名/源码文本级检查; 升级 wheel 后运行可立即发现私有接口
漂移 (清单见 AGENTS.md "依赖版本对接原则", 公开 handler 类的契约由
test_handlers.py 锁定).
"""

import inspect
import unittest

from tests import comfy_stubs

comfy_stubs.install()

from llama_cpp import Llama, _internals  # noqa: E402

from src.core.instruct import DEFAULT_SAMPLING_PARAMS  # noqa: E402


class TestWheelPrivateApiContract(unittest.TestCase):
    def test_llama_close_and_abort(self):
        self.assertTrue(callable(Llama.close))
        self.assertTrue(callable(Llama.abort))

    def test_internals_hybrid_api(self):
        # instruct.is_hybrid_arch 与 hybrid 重置三件套依赖的 C API 封装
        self.assertTrue(callable(_internals.LlamaModel.is_hybrid))
        self.assertTrue(callable(_internals.LlamaModel.is_recurrent))
        self.assertTrue(callable(_internals.LlamaContext.memory_clear))

    def test_instance_attributes_assigned_in_llama_source(self):
        # n_tokens/_ctx/_model/_hybrid_cache_mgr 是实例属性, 无法静态取到,
        # 检查 Llama 类源码中的属性引用作为存在性的近似契约
        source = inspect.getsource(Llama)
        for attr in ("self.n_tokens", "self._ctx", "self._model", "self._hybrid_cache_mgr"):
            self.assertIn(attr, source)

    def test_mmproj_path_instance_attribute_in_handler_source(self):
        # instruct.require_mmproj 经 getattr(handler, "mmproj_path", None) 判定
        # mmproj 是否配置: 属性被 wheel 重命名不会立刻报错, 而是恒判 "未配置"
        # 使全部媒体 Instruct 误报, 静态锁定基类源码中的属性赋值
        import llama_cpp.llama_multimodal as multimodal

        source = inspect.getsource(multimodal.MTMDChatHandler)
        self.assertIn("self.mmproj_path", source)

    def test_mtmd_chat_handler_close(self):
        # storage.clean() 直接调用 chat handler 的 close() 并依赖其幂等性,
        # 兜底 llm 未创建/主模型加载失败路径的 mtmd 资源级联释放
        import llama_cpp.llama_multimodal as multimodal

        self.assertTrue(callable(multimodal.MTMDChatHandler.close))

    def test_ggml_device_symbols(self):
        # devices.py 与 tools/check_devices.py (诊断脚本, 排障工具不能坏在
        # 需要它的时刻) 引用的 _ggml 符号全集
        from llama_cpp import _ggml

        for name in (
            "libggml_base",
            "ggml_backend_dev_count",
            "ggml_backend_dev_get",
            "ggml_backend_load_all_from_path",
            "ggml_backend_reg_count",
            "GGMLBackendDevType",
        ):
            self.assertTrue(hasattr(_ggml, name), f"llama_cpp._ggml missing symbol: {name}")

    def test_split_mode_enum(self):
        import llama_cpp.llama_cpp as llama_cpp_lib

        self.assertTrue(hasattr(llama_cpp_lib.llama_split_mode, "LLAMA_SPLIT_MODE_NONE"))
        self.assertTrue(hasattr(llama_cpp_lib.llama_split_mode, "LLAMA_SPLIT_MODE_LAYER"))

    def test_create_chat_completion_accepts_all_sampling_params(self):
        # Parameters 节点全部字段 + seed + text Instruct allow_thinking 开关折算的
        # reasoning_budget 必须被 create_chat_completion 签名接受;
        # UI 键 max_gen_tokens 按 instruct._run() 的映射折算为 max_tokens 后校验
        params = inspect.signature(Llama.create_chat_completion).parameters
        wheel_names = ["max_tokens" if name == "max_gen_tokens" else name for name in DEFAULT_SAMPLING_PARAMS]
        for name in (*wheel_names, "seed", "reasoning_budget"):
            self.assertIn(name, params, f"create_chat_completion missing param: {name}")


if __name__ == "__main__":
    unittest.main()
