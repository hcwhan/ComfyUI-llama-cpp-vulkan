"""模型生命周期管理: 全局单例存储, 配置校验, 显存折算, ComfyUI 卸载钩子."""

import gc
import os
import sys
import time

import comfy.model_management as mm
from llama_cpp import Llama

from ..i18n.common_static import LOG_PREFIX, NONE_OPTION
from ..i18n.lang import LANG
from ..shared.logger import logger
from .devices import AUTO_LABEL, log_backend_summary, resolve_device_selection
from .gguf_layers import get_model_meta
from .handlers import handler_constructor, is_registered
from .model_paths import get_llm_full_path

_ERRORS = LANG["common"]["storage_errors"]
_LOGS = LANG["logs"]["storage"]

# GGUF 文件体积 -> 运行时显存占用的经验放大系数, 按 n_ctx=8192 校准拆成两项:
# 与上下文无关的计算缓冲等固定开销 + 随 n_ctx 线性增长的 KV/激活开销
# (8192 时合计 1.55). KV 项按体积折算只是元数据不全时的回退:
# KV cache 实际大小与权重量化无关, 强量化模型按体积折算会低估
_BASE_OVERHEAD = 0.15
_KV_OVERHEAD_AT_8K = 0.40
_KV_CALIBRATION_CTX = 8192


def _vram_factor(n_ctx):
    return 1.0 + _BASE_OVERHEAD + _KV_OVERHEAD_AT_8K * n_ctx / _KV_CALIBRATION_CTX


# mmproj(视觉编码器)无 KV cache, 只计固定开销, 不随 n_ctx 增长
_MMPROJ_FACTOR = 1.0 + _BASE_OVERHEAD


def _as_number(value):
    """元数据值归一为数值: 数组(hybrid 模型的逐层 head_count_kv)取均值."""
    if isinstance(value, (list, tuple)):
        return sum(value) / len(value) if value else None
    return value if isinstance(value, (int, float)) else None


def _estimate_kv_bytes(meta, layers, n_ctx):
    """按 GGUF 注意力元数据精确计算 KV cache 字节数(f16), 字段不全时返回 None.

    每 token 每层 KV = head_count_kv * (key_dim + value_dim) * 2 字节;
    hybrid 模型(线性注意力层的 head_count_kv 为 0)经数组均值自然折算.
    """
    kv_heads = _as_number(meta.get("head_count_kv"))
    if not layers or not kv_heads:
        return None
    key_dim = _as_number(meta.get("key_length"))
    if key_dim is None:
        heads = _as_number(meta.get("head_count"))
        embed = _as_number(meta.get("embedding_length"))
        if not heads or not embed:
            return None
        key_dim = embed / heads
    value_dim = _as_number(meta.get("value_length"))
    if value_dim is None:
        value_dim = key_dim
    return int(n_ctx * layers * kv_heads * (key_dim + value_dim) * 2)


def _estimate_per_layer_bytes(model_path, n_ctx):
    """估算每层显存占用(权重+固定开销+KV), 返回 (per_layer_bytes, layers)."""
    meta = get_model_meta(model_path)
    layers = _as_number(meta.get("block_count")) or 32
    size = os.path.getsize(model_path)
    kv_bytes = _estimate_kv_bytes(meta, layers, n_ctx)
    if kv_bytes is None:  # noqa: SIM108 -- 回退分支的注释与两个公式的对照价值高于三元式的紧凑
        # 元数据不全时回退按体积折算的经验系数
        total = size * _vram_factor(n_ctx)
    else:
        total = size * (1.0 + _BASE_OVERHEAD) + kv_bytes
    return total / layers, layers


def _estimate_n_gpu_layers(model_path, mmproj_path, vram_limit, n_ctx):
    """按 GGUF 层数把 vram_limit (GB) 折算成 (n_gpu_layers, mmproj 是否进显存).

    -1 透传给 llama.cpp 的 auto 语义(自动按空闲显存适配, 通常为全部层);
    0 表示纯 CPU 推理; mmproj 只能整只进显存, 体积先从预算中扣除,
    预算连 mmproj 都装不下时两者全留 CPU, 扣除后不足主模型 1 层时
    主模型全留 CPU(mmproj 照常进显存). 全部分支严格遵守 vram_limit 上限,
    层体积为估算值, 实际占用可能略有偏差.
    """
    if vram_limit == -1:
        return -1, mmproj_path is not None
    if vram_limit == 0:
        return 0, False
    per_layer_bytes, _layers = _estimate_per_layer_bytes(model_path, n_ctx)
    layer_size = per_layer_bytes / (1024**3)
    if layer_size <= 0:
        return -1, mmproj_path is not None
    usable = vram_limit
    if mmproj_path:
        mmproj_gb = os.path.getsize(mmproj_path) * _MMPROJ_FACTOR / (1024**3)
        if mmproj_gb >= vram_limit:
            logger.warning(LOG_PREFIX + _LOGS["vram_cannot_fit_mmproj"].format(vram_limit=vram_limit, mmproj_gb=mmproj_gb))
            return 0, False
        usable -= mmproj_gb
    n_layers = int(usable / layer_size)
    if n_layers < 1:
        logger.warning(LOG_PREFIX + _LOGS["vram_no_room_for_layer"].format(vram_limit=vram_limit, layer_size=layer_size))
        return 0, mmproj_path is not None
    return n_layers, mmproj_path is not None


def _estimate_vram_bytes(model_path, mmproj_path, n_gpu_layers, n_ctx):
    """估算本次加载的显存需求(字节), 用于请求 ComfyUI 先腾挪 torch 侧显存."""
    per_layer_bytes, layers = _estimate_per_layer_bytes(model_path, n_ctx)
    gpu_layers = layers if n_gpu_layers < 0 else min(n_gpu_layers, layers)
    size = per_layer_bytes * gpu_layers
    if mmproj_path:
        size += os.path.getsize(mmproj_path) * _MMPROJ_FACTOR
    return int(size)


def resolve_config(config):
    """校验 loader 配置并解析出 (model_path, mmproj_path, handler_cls).

    loader 节点用它做快速失败校验(不实际加载模型),
    load_model 用它取得路径与 handler 类, 两处共享同一套报错.
    handler_cls 已绑定构造期固定参数(注册表 kwargs)与 thinking 开关值
    (handler_constructor 按注册表三态元数据折算, storage 不感知 thinking 逻辑).
    """
    model = config["model"]
    mmproj = config["mmproj"]
    chat_handler = config["chat_handler"]

    model_path = get_llm_full_path(model)
    if model_path is None:
        raise FileNotFoundError(_ERRORS["model_not_found"].format(model=model))

    if chat_handler == NONE_OPTION:
        handler_cls = None
    else:
        thinking = config["thinking"]
        try:
            handler_cls = handler_constructor(chat_handler, thinking)
        except KeyError:
            # 注册过但本构建缺类的 handler 已在启动日志给出 warning,
            # 此处区分两种失配, 避免把 wheel 缺类误报为名字未知
            if is_registered(chat_handler):
                raise ValueError(_ERRORS["handler_unavailable"].format(chat_handler=chat_handler)) from None
            raise ValueError(_ERRORS["unknown_chat_handler"].format(chat_handler=chat_handler)) from None

    mmproj_path = None
    if mmproj and mmproj != NONE_OPTION:
        mmproj_path = get_llm_full_path(mmproj)
        if mmproj_path is None:
            raise FileNotFoundError(_ERRORS["mmproj_not_found"].format(mmproj=mmproj))
        if handler_cls is None:
            raise ValueError(_ERRORS["handler_required_for_mmproj"])
    elif handler_cls is not None:
        # 当前所有 chat handler 均为 VLM handler, 实例化时强制要求 mmproj;
        # 提前拦截, 避免抛出含糊的 "mmproj_path is required"
        raise ValueError(_ERRORS["mmproj_required_for_handler"].format(chat_handler=chat_handler))

    return model_path, mmproj_path, handler_cls


class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None

    @classmethod
    def clean(cls):
        # 未加载(llm/chat_handler 为 None)是常态路径, 显式判空;
        # close 失败不阻断清理, 但留日志供排查显存未释放问题
        if cls.llm is not None:
            try:
                cls.llm.close()
            except Exception as e:
                logger.debug(LOG_PREFIX + _LOGS["llm_close_failed"].format(e=e))

        if cls.chat_handler is not None:
            try:
                # Llama.close() 内部已级联关闭 chat_handler, 此处显式调用依赖
                # close() 的幂等性, 兜底 llm 未创建/主模型加载失败的路径;
                # 用公开的 close()(幂等且完整, mtmd_free + exit_stack)
                cls.chat_handler.close()
            except Exception as e:
                logger.debug(LOG_PREFIX + _LOGS["handler_close_failed"].format(e=e))

        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None

        gc.collect()

    @classmethod
    def load_model(cls, config):
        # 先校验再卸载旧模型: 无效配置不影响当前已加载的模型
        model_path, mmproj_path, handler_cls = resolve_config(config)

        cls.clean()
        gpu_device = config.get("gpu_device", AUTO_LABEL)
        model = config["model"]
        mmproj = config["mmproj"]
        main_gpu, split_mode = resolve_device_selection(gpu_device)

        n_gpu_layers, mmproj_on_gpu = _estimate_n_gpu_layers(model_path, mmproj_path, config["vram_limit"], config["n_ctx"])

        # Vulkan 与 PyTorch 共享同一张物理卡但分配器互不感知, 先请求 ComfyUI
        # 卸载 torch 侧模型腾出显存, 否则 SD 模型占满显存时 Vulkan 分配直接失败;
        # 主模型 0 层时 mmproj 仍可能进显存(use_gpu), 同样需要腾挪.
        # 只腾挪 torch 主设备: 假设 Vulkan 推理与 torch 在同一张卡(单 dGPU 环境
        # 天然成立); 多卡下显式选择其他 Vulkan 卡时, 该卡与 torch 分配器无关,
        # 腾挪主设备无效但也无害
        if n_gpu_layers != 0 or mmproj_on_gpu:
            try:
                mm.free_memory(
                    _estimate_vram_bytes(model_path, mmproj_path if mmproj_on_gpu else None, n_gpu_layers, config["n_ctx"]),
                    mm.get_torch_device(),
                )
            except Exception as e:
                logger.warning(LOG_PREFIX + _LOGS["free_vram_failed"].format(e=e))

        # 加载可达分钟级且中途不可终止, 在真正开始前响应排队期间的取消
        # (点 Cancel 后才轮到本节点执行的场景); 此时旧模型已卸载而新模型
        # 未加载, chat_handler 也未构造, 状态干净
        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        if mmproj_path:
            # handler 构造只校验路径, mmproj 真正加载进显存由 mtmd 在首次推理时
            # 惰性初始化(_init_mtmd_context), 与上面 free_memory 的腾挪同在
            # 一次节点执行内完成, 时序仍有效
            logger.info(LOG_PREFIX + _LOGS["preparing_mmproj"].format(mmproj=mmproj))

            # 注册表固定 kwargs 与 thinking 开关已由 resolve_config 预绑定
            kwargs = {
                "mmproj_path": mmproj_path,
                "verbose": False,
                # <=0 视为未设置, 与库内默认值 -1 语义一致
                "image_min_tokens": config["image_min_tokens"],
                "image_max_tokens": config["image_max_tokens"],
                # vram_limit=0(纯 CPU)或预算装不下 mmproj 时, mmproj(mtmd 编码器)
                # 留在 CPU 以严格遵守显存预算; mtmd 只有 use_gpu 布尔开关,
                # 无法指定设备(见 AGENTS.md 已知问题)
                "use_gpu": mmproj_on_gpu,
            }

            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(_ERRORS["handler_init_failed"].format(e=e)) from e
        else:
            cls.chat_handler = None

        logger.info(LOG_PREFIX + _LOGS["loading_model"].format(model=model))
        logger.info(LOG_PREFIX + _LOGS["load_params"].format(n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, split_mode=split_mode))

        def _create_llama():
            return Llama(
                model_path,
                chat_handler=cls.chat_handler,
                n_gpu_layers=n_gpu_layers,
                main_gpu=main_gpu,
                split_mode=split_mode,
                n_ctx=config["n_ctx"],
                verbose=False,
            )

        try:
            try:
                cls.llm = _create_llama()
            except Exception as e:
                if n_gpu_layers == 0 and not mmproj_on_gpu:
                    raise
                # Windows WDDM 归还显存有延迟, 估算偏低时 Vulkan 分配仍可能失败;
                # 再腾挪一次并稍作等待后重试一轮. 失败原因无法可靠区分是否为
                # 显存不足, 统一重试一次: 非显存错误(文件损坏等)会再次快速失败
                logger.warning(LOG_PREFIX + _LOGS["load_failed_retry"].format(e=e))
                try:
                    mm.free_memory(
                        _estimate_vram_bytes(model_path, mmproj_path if mmproj_on_gpu else None, n_gpu_layers, config["n_ctx"]),
                        mm.get_torch_device(),
                    )
                except Exception as free_err:
                    logger.warning(LOG_PREFIX + _LOGS["free_vram_retry_failed"].format(free_err=free_err))
                time.sleep(1.0)
                cls.llm = _create_llama()
        except Exception:
            # 主模型加载失败时立即回收已创建的 chat_handler, 避免半初始化状态
            # 残留到下一次 load/unload; 此时 mmproj 尚未进显存(mtmd 首次推理时
            # 才惰性加载), close 释放的只是 handler 自身资源
            cls.clean()
            raise
        # 加载成功后才记录配置, 避免加载失败时残留新配置导致后续误判"无需重载"
        cls.current_config = config.copy()
        if n_gpu_layers == 0 and not mmproj_on_gpu:
            # 纯 CPU 推理时打印 Active GPU 会误导排查
            logger.info(LOG_PREFIX + _LOGS["cpu_only"])
        else:
            log_backend_summary(main_gpu, split_mode)


if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models

    def patched_unload_all_models(*args, **kwargs):
        # 补丁按进程生命周期只打一次(hasattr 防重复), 而 "删 sys.modules 再
        # import" 式热重载会产生新模块与新 LLAMA_CPP_STORAGE 类且不重打补丁;
        # 经 sys.modules 动态取当前生效模块的类, 避免旧闭包绑死旧类导致
        # 新模块加载的模型失去 Free 按钮/OOM 清理入口
        module = sys.modules.get(__name__)
        if module is not None:
            module.LLAMA_CPP_STORAGE.clean()
        return mm.unload_all_models_backup(*args, **kwargs)

    mm.unload_all_models = patched_unload_all_models
    logger.info(LOG_PREFIX + _LOGS["cleanup_hook_applied"])
