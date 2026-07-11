import os
import re
import gc
import copy
import threading

import numpy as np

from llama_cpp import Llama

from ..support.cqdm import cqdm
from ..support.gguf_layers import get_layer_count

import comfy.model_management as mm

from .devices import (
    AUTO_LABEL,
    gpu_device_choices,
    resolve_device_selection,
    print_backend_summary,
)
from .handlers import HAS_MTMD, HANDLERS, chat_handler_choices
from .shared import (
    any_type,
    get_llm_filename_list,
    get_llm_full_path,
    image2base64,
    scale_image,
    tensor_to_uint8,
    preset_prompts,
    preset_tags,
)

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


class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            cls.messages.pop(id, None)
            cls.sys_prompts.pop(id, None)

    @classmethod
    def clean(cls, all=False):
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
        if all:
            cls.clean_state()

        gc.collect()

    @classmethod
    def load_model(cls, config):
        cls.clean(all=True)
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        gpu_device = config.get("gpu_device", AUTO_LABEL)
        main_gpu, split_mode = resolve_device_selection(gpu_device)

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

        n_gpu_layers = _estimate_n_gpu_layers(model_path, mmproj_path, config["vram_limit"])

        if mmproj_path:
            print(f"[llama-cpp-vulkan] Loading clip:  {mmproj}")

            # 官方构建只认 clip_model_path;JamePeng 构建把它作为 mmproj_path 的
            # 兼容别名接受(verbose=False 时无告警),统一用旧名兼容两者
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if think_param:
                kwargs[think_param] = "Thinking" in chat_handler
            if HAS_MTMD:
                # <=0 视为未设置,与库内默认值 -1 语义一致;官方构建的旧式
                # handler 不接受这两个参数,跳过
                kwargs["image_max_tokens"] = config["image_max_tokens"]
                kwargs["image_min_tokens"] = config["image_min_tokens"]

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
        # 只卸载模型,保留会话历史(清历史用 llama_cpp_clean_states 节点)
        LLAMA_CPP_STORAGE.clean()
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp-vulkan] Model cleanup hook applied!")


def _is_hybrid_arch(llm):
    """判断模型是否为 hybrid/recurrent 架构(如 Qwen3.5 的线性注意力、Mamba 类)。

    纯 SWA 模型(如 Gemma3)不算:其前缀缓存由 llama-cpp-python 内置的
    checkpoint 机制处理,无需请求后整体重置。
    """
    try:
        return llm._model.is_hybrid() or llm._model.is_recurrent()
    except AttributeError:
        # 官方构建的旧版绑定没有这两个 C API,也不支持 hybrid 模型
        return False


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text):
    """移除 <think>...</think> 推理块。

    Thinking 模型的 generation prompt 通常已注入开头的 <think>,
    此时输出只含闭合标签,需要取最后一个 </think> 之后的内容。
    """
    if "</think>" not in text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned.lstrip()


class _InterruptWatcher:
    """推理期间轮询 ComfyUI 的中断标志,命中时触发 llama 的 abort_event。

    create_completion 在每次请求开始时会 clear abort_event,
    因此命中后持续重复 set 而不是设置一次就退出,避免竞态丢失中断。
    """

    def __init__(self, llm, poll_interval=0.2):
        self.llm = llm
        self.poll_interval = poll_interval
        self.interrupted = False
        self._stop = threading.Event()
        self._thread = None

    def _watch(self):
        while not self._stop.wait(self.poll_interval):
            if mm.processing_interrupted():
                self.interrupted = True
                try:
                    self.llm.abort()
                except Exception:
                    pass

    def __enter__(self):
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        return False


class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = get_llm_filename_list()
        model_list = ["None"] + [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]

        return {"required": {
            "gpu_device": (gpu_device_choices, {
                "default": AUTO_LABEL,
                "tooltip": "Select GPU device for LLM inference.\nAuto = llama.cpp default: prefer dedicated GPU, layer-split across multiple dGPUs.\nSelecting a specific device loads the whole model on that single GPU.\n(iGPU is only selectable when no dGPU exists)"
            }),
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handler_choices, {"default": "None"}),
            "n_ctx": ("INT", {
                "default": 8192,
                "min": 1024, "max": 327680, "step": 128,
                "tooltip": "Context length limit."
            }),
            "vram_limit": ("INT", {
                "default": -1,
                "min": -1, "max": 1024, "step": 1,
                "tooltip": "VRAM usage limit in GB (-1 = no limit)\nReference range; actual usage may slightly exceed."
            }),
            "image_min_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            "image_max_tokens": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            }
        }

    RETURN_TYPES = ("LLAMACPPMODEL",)
    RETURN_NAMES = ("llama_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "llama-cpp-vulkan"

    def loadmodel(self, model, mmproj, chat_handler, gpu_device, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if model == "None":
            raise ValueError("Please select a gguf model.")
        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler": chat_handler,
            "gpu_device": gpu_device,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != custom_config:
            LLAMA_CPP_STORAGE.load_model(custom_config)
        return (custom_config,)


class llama_cpp_instruct_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "preset_prompt": (preset_tags, {"default": preset_tags[1]}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": 'user_prompt\n\nFor preset hints marked with an "*", this will be used to fill the placeholder (e.g., Object names in BBox detection)\nOtherwise, this will override the preset prompts.'}),
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "inference_mode": (["one by one", "images", "video"], {
                    "default": "one by one",
                    "tooltip": "one by one: Read one image at a time\nimages:  \tRead all images at once\nvideo:  \tTreat the input images as video"
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 2,
                    "max": 1024,
                    "step": 1,
                    "tooltip": 'Number of frames to sample evenly from input video.\n(for "video" mode only)'
                }),
                "max_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 16384,
                    "step": 64,
                    "tooltip": 'Max size of input images in "images" and "video" modes.'
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "step": 1, "tooltip": "llama.cpp uses 32-bit seeds; larger values would be truncated."}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
                }),
                "strip_thinking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove <think>...</think> reasoning blocks from the output.\n(for Thinking models)"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "parameters": ("LLAMACPPARAMS",),
                "images": ("IMAGE",),
                "queue_handler": (any_type, {"tooltip": "Used to control the execution order of instruct nodes."}),
            },

        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def sanitize_messages(self, messages):
        clean_messages = copy.deepcopy(messages)
        for msg in clean_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        item["image_url"]["url"] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAsTAAALEwEAmpwYAAAADElEQVQImWP4//8/AAX+Av5Y8msOAAAAAElFTkSuQmCC"
        return clean_messages

    @classmethod
    def IS_CHANGED(cls, save_states=False, **kwargs):
        # save_states 会话依赖节点真实执行来追加历史,输入不变时也不能命中
        # ComfyUI 输出缓存,否则重复提交同一问题不会真的推理
        return float("NaN") if save_states else False

    def _prepare_messages(self, uid, system_prompts, save_states):
        """按 system prompt 变化与 save_states 决定本次请求的初始消息列表。

        返回的是浅拷贝:后续 append 不直接写入存储,推理中断/异常时历史保持一致。
        """
        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(uid)
        if last_sys_prompt != system_prompts:
            # 只清除当前会话,避免误伤其他 state_uid 的历史
            LLAMA_CPP_STORAGE.clean_state(uid)
            LLAMA_CPP_STORAGE.sys_prompts[uid] = system_prompts
            messages = []
        elif save_states:
            print(f"[llama-cpp-vulkan] Loading state and history id={uid}...")
            messages = list(LLAMA_CPP_STORAGE.messages.get(uid, []))
        else:
            messages = []

        # 历史为空时(重)建 system 消息(覆盖 save_states 从 True 切到 False 的场景)
        if not messages and system_prompts.strip():
            messages.append({"role": "system", "content": system_prompts})
        return messages

    def _build_prompt_text(self, preset_prompt, custom_prompt, video_input):
        if custom_prompt.strip() and "*" not in preset_prompt:
            return {"type": "text", "text": custom_prompt}
        if "*" in preset_prompt and not custom_prompt.strip():
            raise ValueError(f'Preset "{preset_prompt}" requires custom_prompt to fill its placeholder (e.g. object categories for BBox detection).')
        # 先替换 @ 再注入用户文本,避免 custom_prompt 中的 @ 被误替换
        p = preset_prompts[preset_prompt].replace("@", "video" if video_input else "image").replace("#", custom_prompt.strip())
        return {"type": "text", "text": p}

    def _infer_one_by_one(self, messages, user_content, frames, seed, params, extract_text, watcher):
        image_content = {"type": "image_url", "image_url": {"url": ""}}
        user_content.append(image_content)
        messages.append({"role": "user", "content": user_content})
        print(f"[llama-cpp-vulkan] Start processing {len(frames)} images")

        out_list = []
        tmp_list = []
        for i, image in enumerate(cqdm(frames)):
            if watcher.interrupted or mm.processing_interrupted():
                raise mm.InterruptProcessingException()
            data = image2base64(tensor_to_uint8(image))
            image_content["image_url"]["url"] = f"data:image/png;base64,{data}"
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **params)
            text = extract_text(output)
            out_list.append(text)
            if len(frames) > 1:
                tmp_list.append(f"====== Image {i+1} ======")
            tmp_list.append(text)

        return "\n\n".join(tmp_list), out_list

    def _run_inference(self, messages, user_content, images, inference_mode, max_frames, max_size, seed, params, extract_text, watcher):
        llm = LLAMA_CPP_STORAGE.llm

        if images is None:
            messages.append({"role": "user", "content": user_content})
            out1 = extract_text(llm.create_chat_completion(messages=messages, seed=seed, **params))
            return out1, [out1]

        # 新版 llama-cpp-python (MTMDChatHandler) 存储 mmproj_path,
        # clip_model_path 仅为旧版本的属性名,两者都要兼容
        handler = LLAMA_CPP_STORAGE.chat_handler
        mmproj_loaded = getattr(handler, "mmproj_path", None) or getattr(handler, "clip_model_path", None)
        if not mmproj_loaded:
            raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")

        frames = images
        if inference_mode == "video":
            # clamp 到实际帧数,避免 linspace 重复采样同一帧浪费上下文
            n_frames = min(max_frames, len(images))
            indices = np.linspace(0, len(images) - 1, n_frames, dtype=int)
            frames = [images[i] for i in indices]

        if inference_mode == "one by one":
            return self._infer_one_by_one(messages, user_content, frames, seed, params, extract_text, watcher)

        # images / video 模式:全部帧并入同一条 user 消息
        for image in frames:
            if len(frames) > 1:
                data = image2base64(scale_image(image, max_size))
            else:
                data = image2base64(tensor_to_uint8(image))
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})

        messages.append({"role": "user", "content": user_content})
        out1 = extract_text(llm.create_chat_completion(messages=messages, seed=seed, **params))
        return out1, [out1]

    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, strip_thinking=True, parameters=None, images=None, queue_handler=None):
        # 校验当前已加载的模型确实是本节点连线的配置:
        # 多组 loader+instruct 交错执行时,全局单例可能已被切换成其他模型
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model)

        # 先复制再修改,避免污染 ComfyUI 缓存的共享参数 dict;
        # present_penalty 在当前 llama-cpp-python (>=0.3.41) 中受支持,不再丢弃
        _parameters = (parameters or {}).copy()
        _uid = _parameters.pop("state_uid", None)
        # 转 int 保证 state_uid 输出与声明的 INT 类型一致(unique_id 是数字字符串)
        uid = int(unique_id.rpartition('.')[-1]) if _uid in (None, -1) else _uid

        video_input = inference_mode == "video"
        # 英文指令对多语言模型的跟随更稳定
        system_prompts = "Treat the input image sequence as a continuous video rather than independent still frames. " + system_prompt if video_input else system_prompt

        messages = self._prepare_messages(uid, system_prompts, save_states)
        user_content = [self._build_prompt_text(preset_prompt, custom_prompt, video_input)]

        def extract_text(output):
            text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            return _strip_thinking(text) if strip_thinking else text

        # 监视线程让长时间生成也能响应 ComfyUI 的取消操作
        with _InterruptWatcher(LLAMA_CPP_STORAGE.llm) as watcher:
            out1, out2 = self._run_inference(
                messages, user_content, images, inference_mode,
                max_frames, max_size, seed, _parameters, extract_text, watcher,
            )

        if watcher.interrupted:
            # abort_event 使生成提前返回了截断结果,丢弃并走标准中断流程
            raise mm.InterruptProcessingException()

        if save_states:
            print(f"[llama-cpp-vulkan] Saving state id={uid}...")
            messages.append({"role": "assistant", "content": out1})
            LLAMA_CPP_STORAGE.messages[uid] = self.sanitize_messages(messages)
        elif not LLAMA_CPP_STORAGE.messages.get(uid):
            LLAMA_CPP_STORAGE.sys_prompts.pop(uid, None)

        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        elif _is_hybrid_arch(LLAMA_CPP_STORAGE.llm):
            # 真 hybrid/recurrent 架构(Qwen3.5、LFM2 系等)的线性注意力状态无法
            # 跨请求做前缀复用,不重置会导致后续请求输出错乱;按架构判断而非
            # handler 名单,避免每加一个 hybrid 模型都要维护名单
            llm = LLAMA_CPP_STORAGE.llm
            llm.n_tokens = 0
            llm._ctx.memory_clear(True)
            if llm._hybrid_cache_mgr is not None:
                llm._hybrid_cache_mgr.clear()

        return (out1, out2, uid)


class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1, "tooltip": "Max tokens to generate (0 = unlimited, bounded by n_ctx)."}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "present_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "mirostat_mode": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Use a specific ID to save the conversation state.\n(-1 = use node's unique_id)"
                }),
            }
        }
    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"
    def process(self, **kwargs):
        return (kwargs,)


class llama_cpp_clean_states:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type,),
                "state_uid": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "Clear the saved state for a specific ID (-1 = clear all)"
                }),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, any, state_uid):
        print(f"[llama-cpp-vulkan] Cleaning up saved states {state_uid}...")
        LLAMA_CPP_STORAGE.clean_state(state_uid)
        return (any,)


class llama_cpp_unload_model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any": (any_type,)}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, any):
        print("[llama-cpp-vulkan] Unloading llama model...")
        LLAMA_CPP_STORAGE.clean()
        return (any,)
