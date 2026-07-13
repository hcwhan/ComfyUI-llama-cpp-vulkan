"""模型生命周期管理: 全局单例存储, 配置校验, 显存折算, ComfyUI 卸载钩子."""

import os
import gc

from llama_cpp import Llama

import comfy.model_management as mm

from ..shared.logger import logger
from .devices import AUTO_LABEL, resolve_device_selection, log_backend_summary
from .gguf_layers import get_layer_count
from .handlers import HANDLERS
from .model_paths import get_llm_full_path

# GGUF 文件体积 -> 运行时显存占用的经验放大系数,按 n_ctx=8192 校准拆成两项:
# 与上下文无关的计算缓冲等固定开销 + 随 n_ctx 线性增长的 KV/激活开销
# (8192 时合计 1.55)
_BASE_OVERHEAD = 0.15
_KV_OVERHEAD_AT_8K = 0.40
_KV_CALIBRATION_CTX = 8192


def _vram_factor(n_ctx):
    return 1.0 + _BASE_OVERHEAD + _KV_OVERHEAD_AT_8K * n_ctx / _KV_CALIBRATION_CTX


# mmproj(视觉编码器)无 KV cache,只计固定开销,不随 n_ctx 增长
_MMPROJ_FACTOR = 1.0 + _BASE_OVERHEAD


def _estimate_n_gpu_layers(model_path, mmproj_path, vram_limit, n_ctx):
    """按 GGUF 层数把 vram_limit (GB) 折算成 n_gpu_layers。

    -1 透传给 llama.cpp 的 auto 语义(自动按空闲显存适配,通常为全部层);
    0 表示纯 CPU 推理;mmproj 常驻显存,先从预算中扣除。
    """
    if vram_limit == -1:
        return -1
    if vram_limit == 0:
        return 0
    factor = _vram_factor(n_ctx)
    layers = get_layer_count(model_path) or 32
    layer_size = os.path.getsize(model_path) * factor / (1024 ** 3) / layers
    if layer_size <= 0:
        return -1
    usable = vram_limit
    if mmproj_path:
        usable -= os.path.getsize(mmproj_path) * _MMPROJ_FACTOR / (1024 ** 3)
    if usable <= 0:
        # mmproj 只能整只进显存,预算无法满足时主模型全留 CPU 以免雪上加霜
        logger.warning(f"[llama-cpp-vulkan] vram_limit ({vram_limit} GB) is fully consumed by the mmproj file, keeping all main model layers on CPU")
        return 0
    return max(1, int(usable / layer_size))


def _estimate_vram_bytes(model_path, mmproj_path, n_gpu_layers, n_ctx):
    """估算本次加载的显存需求(字节),用于请求 ComfyUI 先腾挪 torch 侧显存。"""
    factor = _vram_factor(n_ctx)
    size = os.path.getsize(model_path) * factor
    if n_gpu_layers >= 0:
        layers = get_layer_count(model_path) or 32
        size *= min(1.0, n_gpu_layers / layers)
    if mmproj_path:
        size += os.path.getsize(mmproj_path) * _MMPROJ_FACTOR
    return int(size)


def resolve_config(config):
    """校验 loader 配置并解析出 (model_path, mmproj_path, handler_cls)。

    loader 节点用它做快速失败校验(不实际加载模型),
    load_model 用它取得路径与 handler 类,两处共享同一套报错。
    handler_cls 已由注册表预绑定构造期固定参数(thinking 开关等)。
    """
    model = config["model"]
    mmproj = config["mmproj"]
    chat_handler = config["chat_handler"]

    model_path = get_llm_full_path(model)
    if model_path is None:
        raise FileNotFoundError(f"Model '{model}' not found in any llm/LLM folder")

    if chat_handler == "None":
        handler_cls = None
    else:
        try:
            handler_cls = HANDLERS[chat_handler]
        except KeyError:
            raise ValueError(f'Unknown chat handler: "{chat_handler}"') from None

    mmproj_path = None
    if mmproj and mmproj != "None":
        mmproj_path = get_llm_full_path(mmproj)
        if mmproj_path is None:
            raise FileNotFoundError(f"mmproj '{mmproj}' not found in any llm/LLM folder")
        if handler_cls is None:
            raise ValueError("Please select a chat handler for vision model.")
    elif handler_cls is not None:
        # 当前所有 chat handler 均为 VLM handler,实例化时强制要求 mmproj;
        # 提前拦截,避免抛出含糊的 "mmproj_path is required"
        raise ValueError(
            f'Chat handler "{chat_handler}" requires a mmproj model. '
            'Select the matching mmproj file, or set chat_handler to "None" for text-only models.'
        )

    return model_path, mmproj_path, handler_cls


class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None

    @classmethod
    def clean(cls):
        # 未加载(llm/chat_handler 为 None)是常态路径,显式判空;
        # close 失败不阻断清理,但留日志供排查显存未释放问题
        if cls.llm is not None:
            try:
                cls.llm.close()
            except Exception as e:
                logger.debug(f"[llama-cpp-vulkan] llm close failed: {e}")

        if cls.chat_handler is not None:
            try:
                # Llama.close() 内部已级联关闭 chat_handler,此处显式调用依赖
                # close() 的幂等性,兜底 llm 未创建/主模型加载失败的路径;
                # 用公开的 close()(幂等且完整,mtmd_free + exit_stack)
                cls.chat_handler.close()
            except Exception as e:
                logger.debug(f"[llama-cpp-vulkan] chat_handler close failed: {e}")

        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None

        gc.collect()

    @classmethod
    def load_model(cls, config):
        # 先校验再卸载旧模型:无效配置不影响当前已加载的模型
        model_path, mmproj_path, handler_cls = resolve_config(config)

        cls.clean()
        model = config["model"]
        mmproj = config["mmproj"]
        gpu_device = config.get("gpu_device", AUTO_LABEL)
        main_gpu, split_mode = resolve_device_selection(gpu_device)

        n_gpu_layers = _estimate_n_gpu_layers(model_path, mmproj_path, config["vram_limit"], config["n_ctx"])

        # Vulkan 与 PyTorch 共享同一张物理卡但分配器互不感知,先请求 ComfyUI
        # 卸载 torch 侧模型腾出显存,否则 SD 模型占满显存时 Vulkan 分配直接失败;
        # 主模型 0 层时 mmproj 仍可能进显存(use_gpu),同样需要腾挪。
        # 只腾挪 torch 主设备:假设 Vulkan 推理与 torch 在同一张卡(单 dGPU 环境
        # 天然成立);多卡下显式选择其他 Vulkan 卡时,该卡与 torch 分配器无关,
        # 腾挪主设备无效但也无害
        mmproj_on_gpu = mmproj_path is not None and config["vram_limit"] != 0
        if n_gpu_layers != 0 or mmproj_on_gpu:
            try:
                mm.free_memory(
                    _estimate_vram_bytes(model_path, mmproj_path, n_gpu_layers, config["n_ctx"]),
                    mm.get_torch_device(),
                )
            except Exception as e:
                logger.warning(f"[llama-cpp-vulkan] failed to free torch VRAM before load: {e}")

        if mmproj_path:
            logger.info(f"[llama-cpp-vulkan] Loading mmproj: {mmproj}")

            # thinking 开关等构造期固定参数已由注册表经 functools.partial 预绑定
            kwargs = {
                "mmproj_path": mmproj_path,
                "verbose": False,
                # <=0 视为未设置,与库内默认值 -1 语义一致
                "image_max_tokens": config["image_max_tokens"],
                "image_min_tokens": config["image_min_tokens"],
                # vram_limit=0 表示纯 CPU 推理,mmproj(mtmd 编码器)同样留在 CPU;
                # mtmd 只有 use_gpu 布尔开关,无法指定设备(见 AGENTS.md 已知问题)
                "use_gpu": config["vram_limit"] != 0,
            }

            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nChatHandler initialization failed. Please update llama-cpp-python to the latest version with Vulkan support.")
        else:
            cls.chat_handler = None

        logger.info(f"[llama-cpp-vulkan] Loading model: {model}")
        logger.info(f"[llama-cpp-vulkan] n_gpu_layers = {n_gpu_layers}, main_gpu = {main_gpu}, split_mode = {split_mode}")
        try:
            cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, split_mode=split_mode, n_ctx=config["n_ctx"], verbose=False)
        except Exception:
            # 主模型加载失败时立即释放已创建的 chat_handler(mmproj 已进显存),
            # 不等下一次 load/unload 的 clean() 才回收
            cls.clean()
            raise
        # 加载成功后才记录配置,避免加载失败时残留新配置导致后续误判"无需重载"
        cls.current_config = config.copy()
        if n_gpu_layers == 0 and not mmproj_on_gpu:
            # 纯 CPU 推理时打印 Active GPU 会误导排查
            logger.info("[llama-cpp-vulkan] CPU-only inference: no layers or mmproj offloaded to GPU")
        else:
            log_backend_summary(main_gpu, split_mode)


if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean()
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    logger.info("[llama-cpp-vulkan] Model cleanup hook applied!")
