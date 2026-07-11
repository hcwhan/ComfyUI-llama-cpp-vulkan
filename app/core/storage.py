"""模型生命周期管理: 全局单例存储, 配置校验, 显存折算, ComfyUI 卸载钩子."""

import os
import gc

from llama_cpp import Llama

import comfy.model_management as mm

from .devices import AUTO_LABEL, resolve_device_selection, print_backend_summary
from .gguf_layers import get_layer_count
from .handlers import HANDLERS
from .model_paths import get_llm_full_path

# GGUF 文件体积 -> 运行时显存占用的经验放大系数(权重 + KV/计算缓冲)
_VRAM_OVERHEAD_FACTOR = 1.55


def _estimate_n_gpu_layers(model_path, mmproj_path, vram_limit):
    """按 GGUF 层数把 vram_limit (GB) 折算成 n_gpu_layers。

    -1 表示不限制,全部层上 GPU;mmproj 常驻显存,先从预算中扣除。
    """
    if vram_limit == -1:
        return -1
    layers = get_layer_count(model_path) or 32
    layer_size = os.path.getsize(model_path) * _VRAM_OVERHEAD_FACTOR / (1024 ** 3) / layers
    usable = vram_limit
    if mmproj_path:
        usable -= os.path.getsize(mmproj_path) * _VRAM_OVERHEAD_FACTOR / (1024 ** 3)
    return max(1, int(usable / layer_size))


def resolve_config(config):
    """校验 loader 配置并解析出 (model_path, mmproj_path, handler_cls, think_param)。

    loader 节点用它做快速失败校验(不实际加载模型),
    load_model 用它取得路径与 handler 类,两处共享同一套报错。
    """
    model = config["model"]
    mmproj = config["mmproj"]
    chat_handler = config["chat_handler"]

    model_path = get_llm_full_path(model)
    if model_path is None:
        raise FileNotFoundError(f"Model '{model}' not found in any llm/LLM folder")

    if chat_handler == "None":
        handler_cls = think_param = None
    else:
        try:
            handler_cls, think_param = HANDLERS[chat_handler]
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

    return model_path, mmproj_path, handler_cls, think_param


class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None

    @classmethod
    def clean(cls):
        try:
            cls.llm.close()
        except Exception:
            pass

        try:
            # 公开的 close() 幂等且完整(mtmd_free + exit_stack);
            # 直接调 _exit_stack.close() 会跳过 mtmd 视觉编码器的释放
            cls.chat_handler.close()
        except Exception:
            pass

        cls.llm = None
        cls.chat_handler = None
        cls.current_config = None

        gc.collect()

    @classmethod
    def load_model(cls, config):
        # 先校验再卸载旧模型:无效配置不影响当前已加载的模型
        model_path, mmproj_path, handler_cls, think_param = resolve_config(config)

        cls.clean()
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        gpu_device = config.get("gpu_device", AUTO_LABEL)
        main_gpu, split_mode = resolve_device_selection(gpu_device)

        n_gpu_layers = _estimate_n_gpu_layers(model_path, mmproj_path, config["vram_limit"])

        if mmproj_path:
            print(f"[llama-cpp-vulkan] Loading clip:  {mmproj}")

            kwargs = {
                "mmproj_path": mmproj_path,
                "verbose": False,
                # <=0 视为未设置,与库内默认值 -1 语义一致
                "image_max_tokens": config["image_max_tokens"],
                "image_min_tokens": config["image_min_tokens"],
            }
            if think_param:
                kwargs[think_param] = "Thinking" in chat_handler

            try:
                cls.chat_handler = handler_cls(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nChatHandler initialization failed. Please update llama-cpp-python to the latest version with Vulkan support.")
        else:
            cls.chat_handler = None

        print(f"[llama-cpp-vulkan] Loading model: {model}")
        print(f"[llama-cpp-vulkan] n_gpu_layers = {n_gpu_layers}, main_gpu = {main_gpu}, split_mode = {split_mode}")
        cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, split_mode=split_mode, n_ctx=config["n_ctx"], verbose=False)
        # 加载成功后才记录配置,避免加载失败时残留新配置导致后续误判"无需重载"
        cls.current_config = config.copy()
        print_backend_summary(main_gpu, split_mode)


if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean()
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp-vulkan] Model cleanup hook applied!")
