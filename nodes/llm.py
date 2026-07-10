import os
import gc
import copy
import json
import ctypes
from pathlib import Path

import numpy as np
import torch

from llama_cpp import Llama
import llama_cpp.llama_cpp as _llama_cpp_lib
from llama_cpp._ggml import (
    libggml_base,
    ggml_backend_dev_count,
    ggml_backend_dev_get,
    ggml_backend_load_all_from_path,
)

from ..support.cqdm import cqdm
from ..support.gguf_layers import get_layer_count

import comfy.model_management as mm

from .shared import (
    any_type,
    get_llm_filename_list,
    get_llm_full_path,
    image2base64,
    scale_image,
    preset_prompts,
    preset_tags,
)

libggml_base.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_name.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_description.restype = ctypes.c_char_p
libggml_base.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
libggml_base.ggml_backend_dev_type.restype = ctypes.c_int32

_GGML_BACKEND_DEVICE_TYPE_GPU = 1
_GGML_BACKEND_DEVICE_TYPE_IGPU = 2
_DEV_TYPE_NAMES = {1: "GPU", 2: "IGPU"}


def _detect_gpu_devices():
    try:
        lib_dir = Path(_llama_cpp_lib.__file__).resolve().parent / "lib"
        if not lib_dir.exists():
            return []
        ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))

        devices = []
        for i in range(ggml_backend_dev_count()):
            dev = ggml_backend_dev_get(i)
            dev_type = libggml_base.ggml_backend_dev_type(dev)
            if dev_type in (_GGML_BACKEND_DEVICE_TYPE_GPU, _GGML_BACKEND_DEVICE_TYPE_IGPU):
                name = libggml_base.ggml_backend_dev_name(dev).decode("utf-8", errors="replace")
                desc = libggml_base.ggml_backend_dev_description(dev).decode("utf-8", errors="replace").strip()
                type_name = _DEV_TYPE_NAMES.get(dev_type, "GPU")
                devices.append({"name": name, "desc": desc, "type": type_name})
        return devices
    except Exception as e:
        print(f"[llama-cpp-vulkan] WARNING: GPU detection failed: {e}")
        return []


_gpu_devices = _detect_gpu_devices()

if _gpu_devices:
    _summary = ", ".join(f"{d['name']} ({d['desc']}) [{d['type']}]" for d in _gpu_devices)
    print(f"[llama-cpp-vulkan] Detected {len(_gpu_devices)} GPU device(s): {_summary}")
else:
    print("[llama-cpp-vulkan] WARNING: No GPU devices detected, running on CPU only")


_AUTO_LABEL = "Auto (独显优先)"


def _build_gpu_device_choices():
    choices = [_AUTO_LABEL]
    gpu_first = sorted(_gpu_devices, key=lambda d: (d["type"] != "GPU", d["name"]))
    for dev in gpu_first:
        choices.append(f"{dev['name']} - {dev['desc']} [{dev['type']}]")
    return choices


def _resolve_main_gpu(gpu_device):
    if gpu_device == _AUTO_LABEL:
        for i, dev in enumerate(_gpu_devices):
            if dev["type"] == "GPU":
                return i
        return 0
    for i, dev in enumerate(_gpu_devices):
        label = f"{dev['name']} - {dev['desc']} [{dev['type']}]"
        if label == gpu_device:
            return i
    return 0


_gpu_device_choices = _build_gpu_device_choices()


def _print_backend_summary(main_gpu=0):
    try:
        if _gpu_devices:
            active = _gpu_devices[main_gpu] if main_gpu < len(_gpu_devices) else _gpu_devices[0]
            print(f"[llama-cpp-vulkan] Active GPU: {active['name']} ({active['desc']}) [{active['type']}]")
        else:
            print("[llama-cpp-vulkan] WARNING: No GPU backend detected, running on CPU only")
    except Exception:
        pass


from llama_cpp.llama_chat_format import (
    Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler,
    NanoLlavaChatHandler, Llama3VisionAlphaChatHandler, MiniCPMv26ChatHandler
)

chat_handlers = ["None", "LLaVA-1.5", "LLaVA-1.6", "Moondream2", "nanoLLaVA", "llama3-Vision-Alpha", "MiniCPM-v2.6"]

try:
    from llama_cpp.llama_chat_format import MTMDChatHandler
    chat_handlers += ["DeepSeek-OCR"]
    _MTMD = True
except:
    _MTMD = False

try:
    from llama_cpp.llama_chat_format import Gemma3ChatHandler
    chat_handlers += ["Gemma3"]
except:
    Gemma3ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Gemma4ChatHandler
    chat_handlers += ["Gemma4"]
except:
    Gemma4ChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    chat_handlers += ["Qwen2.5-VL", "MinerU2.5-Pro"]
except:
    Qwen25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
    chat_handlers += ["Qwen3-VL", "Qwen3-VL-Thinking"]
except:
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
    chat_handlers += ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]
except:
    Qwen35ChatHandler = None

try:
    from llama_cpp.llama_chat_format import (GLM46VChatHandler, LFM2VLChatHandler, GLM41VChatHandler)
    chat_handlers += ["GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking", "LFM2-VL"]
except:
    GLM46VChatHandler = None
    LFM2VLChatHandler = None
    GLM41VChatHandler = None

try:
    from llama_cpp.llama_chat_format import LFM25VLChatHandler
    chat_handlers += ["LFM2.5-VL"]
except:
    LFM25VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
    chat_handlers += ["Granite-Docling"]
except:
    GraniteDoclingChatHandler = None

try:
    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
    chat_handlers += ["MiniCPM-v4.5", "MiniCPM-v4.5-Thinking"]
except:
    MiniCPMv45ChatHandler = None

try:
    from llama_cpp.llama_chat_format import MiniCPMv46ChatHandler
    chat_handlers += ["MiniCPM-v4.6", "MiniCPM-v4.6-Thinking"]
except:
    MiniCPMv46ChatHandler = None

try:
    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
    chat_handlers += ["PaddleOCR-VL-1.5"]
except:
    PaddleOCRChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen3ASRChatHandler
    chat_handlers += ["Qwen3-ASR"]
except:
    Qwen3ASRChatHandler = None

try:
    from llama_cpp.llama_chat_format import Step3VLChatHandler
    chat_handlers += ["Step3-VL"]
except:
    Step3VLChatHandler = None


class LLAMA_CPP_STORAGE:
    llm = None
    chat_handler = None
    current_config = None
    #states = {}
    messages = {}
    sys_prompts = {}

    @classmethod
    def clean_state(cls, id=-1):
        if id == -1:
            #cls.states.clear()
            cls.messages.clear()
            cls.sys_prompts.clear()
        else:
            #cls.states.pop(f"{id}", None)
            cls.messages.pop(f"{id}", None)
            cls.sys_prompts.pop(f"{id}", None)

    @classmethod
    def clean(cls, all=False):
        try:
            cls.llm.close()
        except Exception:
            pass

        try:
            # 公开的 close() 幂等且完整（mtmd_free + exit_stack）；
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
        def get_chat_handler(chat_handler):
            match chat_handler:
                case "Qwen3.5"|"Qwen3.5-Thinking"|"Qwen3.6"|"Qwen3.6-Thinking":
                    return Qwen35ChatHandler
                case "Qwen3-VL"|"Qwen3-VL-Thinking":
                    return Qwen3VLChatHandler
                case "Qwen3-ASR":
                    return Qwen3ASRChatHandler
                case "Qwen2.5-VL"|"MinerU2.5-Pro":
                    return Qwen25VLChatHandler
                case "LLaVA-1.5":
                    return Llava15ChatHandler
                case "LLaVA-1.6":
                    return Llava16ChatHandler
                case "Moondream2":
                    return MoondreamChatHandler
                case "nanoLLaVA":
                    return NanoLlavaChatHandler
                case "llama3-Vision-Alpha":
                    return Llama3VisionAlphaChatHandler
                case "MiniCPM-v2.6":
                    return MiniCPMv26ChatHandler
                case "MiniCPM-v4.5"|"MiniCPM-v4.5-Thinking":
                    return MiniCPMv45ChatHandler
                case "MiniCPM-v4.6"|"MiniCPM-v4.6-Thinking":
                    return MiniCPMv46ChatHandler
                case "Gemma3":
                    return Gemma3ChatHandler
                case "Gemma4":
                    return Gemma4ChatHandler
                case "GLM-4.6V"|"GLM-4.6V-Thinking":
                    return GLM46VChatHandler
                case "GLM-4.1V-Thinking":
                    return GLM41VChatHandler
                case "LFM2-VL":
                    return LFM2VLChatHandler
                case "LFM2.5-VL":
                    return LFM25VLChatHandler
                case "Granite-Docling":
                    return GraniteDoclingChatHandler
                case "DeepSeek-OCR":
                    return MTMDChatHandler
                case "PaddleOCR-VL-1.5":
                    return PaddleOCRChatHandler
                case "Step3-VL":
                    return Step3VLChatHandler
                case "None":
                    return None
                case _:
                    raise ValueError(f'Unknown model type: "{chat_handler}"')

        cls.clean(all=True)
        model = config["model"]
        mmproj = config["mmproj"]
        chat_handler = config["chat_handler"]
        gpu_device = config.get("gpu_device", _AUTO_LABEL)
        n_ctx = config["n_ctx"]
        vram_limit = config["vram_limit"]
        image_max_tokens = config["image_max_tokens"]
        image_min_tokens = config["image_min_tokens"]
        main_gpu = _resolve_main_gpu(gpu_device)
        n_gpu_layers = -1

        model_path = get_llm_full_path(model)
        if model_path is None:
            raise FileNotFoundError(f"Model '{model}' not found in any llm/LLM folder")
        handler = get_chat_handler(chat_handler)

        if vram_limit != -1:
            gguf_layers = get_layer_count(model_path) or 32
            gguf_size = os.path.getsize(model_path)  * 1.55 / (1024 ** 3)
            gguf_layer_size = gguf_size / gguf_layers

        if mmproj and mmproj != "None":
            mmproj_path = get_llm_full_path(mmproj)
            if mmproj_path is None:
                raise FileNotFoundError(f"mmproj '{mmproj}' not found in any llm/LLM folder")
            if chat_handler == "None":
                raise ValueError("Please select a chat handler for vision model.")

            if vram_limit != -1:
                mmproj_size = os.path.getsize(mmproj_path)  * 1.55 / (1024 ** 3)
                n_gpu_layers = max(1, int((vram_limit - mmproj_size) / gguf_layer_size))

            print(f"[llama-cpp-vulkan] Loading clip:  {mmproj}")

            think_mode = "Thinking" in chat_handler
            kwargs = {"clip_model_path": mmproj_path, "verbose": False}
            if chat_handler in ["Qwen3-VL", "Qwen3-VL-Thinking"]:
                kwargs["force_reasoning"] = think_mode
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens
            elif chat_handler in [
                "MiniCPM-v4.5", "MiniCPM-v4.5-Thinking",
                "MiniCPM-v4.6", "MiniCPM-v4.6-Thinking",
                "GLM-4.6V", "GLM-4.6V-Thinking", "GLM-4.1V-Thinking",
                "Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking",
            ]:
                kwargs["enable_thinking"] = think_mode

            if _MTMD:
                kwargs["image_max_tokens"] = image_max_tokens
                kwargs["image_min_tokens"] = image_min_tokens

            try:
                cls.chat_handler = handler(**kwargs)
            except Exception as e:
                raise RuntimeError(f"{e}\nChatHandler initialization failed. Please update llama-cpp-python to the latest version with Vulkan support.")

        else:
            if vram_limit != -1:
                n_gpu_layers = max(1, int(vram_limit / gguf_layer_size))
            if handler is not None:
                cls.chat_handler = handler(verbose=False)
            else:
                cls.chat_handler = None

        print(f"[llama-cpp-vulkan] Loading model: {model}")
        print(f"[llama-cpp-vulkan] n_gpu_layers = {n_gpu_layers}, main_gpu = {main_gpu}")
        cls.llm = Llama(model_path, chat_handler=cls.chat_handler, n_gpu_layers=n_gpu_layers, main_gpu=main_gpu, n_ctx=n_ctx, verbose=False)
        # 加载成功后才记录配置，避免加载失败时残留新配置导致后续误判"无需重载"
        cls.current_config = config.copy()
        _print_backend_summary(main_gpu)


if not hasattr(mm, "unload_all_models_backup"):
    mm.unload_all_models_backup = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        LLAMA_CPP_STORAGE.clean(all=True)
        result = mm.unload_all_models_backup(*args, **kwargs)
        return result
    mm.unload_all_models = patched_unload_all_models
    print("[llama-cpp-vulkan] Model cleanup hook applied!")


class llama_cpp_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        all_llms = get_llm_filename_list()
        model_list = ["None"] + [f for f in all_llms if "mmproj" not in f.lower()]
        mmproj_list = ["None"] + [f for f in all_llms if "mmproj" in f.lower()]

        return {"required": {
            "gpu_device": (_gpu_device_choices, {
                "default": _AUTO_LABEL,
                "tooltip": "Select GPU device for LLM inference.\nAuto = prefer dedicated GPU over integrated GPU."
            }),
            "model": (model_list,),
            "mmproj": (mmproj_list, {"default": "None"}),
            "chat_handler": (chat_handlers, {"default": "None"}),
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

    '''
    @classmethod
    def IS_CHANGED(s, model, mmproj, chat_handler, n_ctx, vram_limit, image_min_tokens, image_max_tokens):
        if LLAMA_CPP_STORAGE.llm is None:
            return float("NaN")

        custom_config = {
            "model": model,
            "mmproj": mmproj,
            "chat_handler":chat_handler,
            "n_ctx": n_ctx,
            "vram_limit": vram_limit,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens
        }
        config_str = json.dumps(custom_config, sort_keys=True, ensure_ascii=False)
        return config_str
    '''
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
            print("[llama-cpp-vulkan] Loading model...")
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload the model after inference."
                }),
                "save_states": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Preserve the context of this conversation in RAM."
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

    def process(self, llama_model, preset_prompt, custom_prompt, system_prompt, inference_mode, max_frames, max_size, seed, force_offload, save_states, unique_id, parameters=None, images=None, queue_handler=None):
        # 校验当前已加载的模型确实是本节点连线的配置：
        # 多组 loader+instruct 交错执行时，全局单例可能已被切换成其他模型
        if not LLAMA_CPP_STORAGE.llm or LLAMA_CPP_STORAGE.current_config != llama_model:
            LLAMA_CPP_STORAGE.load_model(llama_model)
            #raise RuntimeError("The model has been unloaded or failed to load!")

        if parameters is None:
            parameters = {}

        # 先复制再修改，避免污染 ComfyUI 缓存的共享参数 dict；
        # present_penalty 在当前 llama-cpp-python (>=0.3.41) 中受支持，不再丢弃
        _parameters = parameters.copy()
        _uid = _parameters.pop("state_uid", None)
        uid = unique_id.rpartition('.')[-1] if _uid in (None, -1) else _uid

        last_sys_prompt = LLAMA_CPP_STORAGE.sys_prompts.get(f"{uid}", None)
        video_input = inference_mode == "video"
        system_prompts = "请将输入的图片序列当做视频而不是静态帧序列, " + system_prompt if video_input else system_prompt
        if last_sys_prompt != system_prompts:
            messages = []
            LLAMA_CPP_STORAGE.clean_state()
            LLAMA_CPP_STORAGE.sys_prompts[f"{uid}"] = system_prompts
            if system_prompts.strip():
                messages.append({"role": "system", "content": system_prompts})
        else:
            if save_states:
                try:
                    print(f"[llama-cpp-vulkan] Loading state and history id={uid}...")
                    #LLAMA_CPP_STORAGE.llm.load_state(LLAMA_CPP_STORAGE.states[f"{uid}"])
                    messages = LLAMA_CPP_STORAGE.messages.get(f"{uid}", [])
                except Exception as e:
                    messages = []
            else:
                messages = []
        out1 = ""
        out2 = []
        user_content = []
        if custom_prompt.strip() and "*" not in preset_prompt:
            user_content.append({"type": "text", "text": custom_prompt})
        else:
            p = preset_prompts[preset_prompt].replace("#", custom_prompt.strip()).replace("@", "video" if video_input else "image")
            user_content.append({"type": "text", "text": p})

        if images is not None:
            if not hasattr(LLAMA_CPP_STORAGE.chat_handler, "clip_model_path") or LLAMA_CPP_STORAGE.chat_handler.clip_model_path is None:
                 raise ValueError("Image input detected, but the loaded model is not configured with a mmproj module.")

            frames = images
            if video_input:
                indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
                frames = [images[i] for i in indices]

            if inference_mode == "one by one":
                tmp_list = []
                image_content = {
                    "type": "image_url",
                    "image_url": {"url": ""}
                }
                user_content.append(image_content)
                messages.append({"role": "user", "content": user_content})
                print(f"[llama-cpp-vulkan] Start processing {len(frames)} images")

                for i, image in enumerate(cqdm(frames)):
                    if mm.processing_interrupted():
                        raise mm.InterruptProcessingException()
                    data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    for item in user_content:
                        if item.get("type") == "image_url":
                            item["image_url"]["url"] = f"data:image/jpeg;base64,{data}"
                            break
                    output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                    text = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                    out2.append(text)
                    if len(frames) > 1:
                        tmp_list.append(f"====== Image {i+1} ======")
                    tmp_list.append(text)

                out1 = "\n\n".join(tmp_list)
            else:
                for image in frames:
                    if len(frames) > 1:
                        data = image2base64(scale_image(image, max_size))
                    else:
                        data = image2base64(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{data}"}
                    }
                    user_content.append(image_content)

                messages.append({"role": "user", "content": user_content})
                output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
                out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
                out2 = [out1]
        else:
            messages.append({"role": "user", "content": user_content})
            output = LLAMA_CPP_STORAGE.llm.create_chat_completion(messages=messages, seed=seed, **_parameters)
            out1 = output['choices'][0]['message']['content'].removeprefix(": ").lstrip()
            out2 = [out1]

        if save_states:
            print(f"[llama-cpp-vulkan] Saving state id={uid}...")
            #LLAMA_CPP_STORAGE.states[f"{uid}"] = LLAMA_CPP_STORAGE.llm.save_state()
            messages.append({"role": "assistant", "content": out1})
            clear_message = self.sanitize_messages(messages)
            LLAMA_CPP_STORAGE.messages[f"{uid}"] = clear_message
        else:
            if not LLAMA_CPP_STORAGE.messages.get(f"{uid}"):
                LLAMA_CPP_STORAGE.sys_prompts.pop(f"{uid}", None)

        if force_offload:
            LLAMA_CPP_STORAGE.clean()
        else:
            if LLAMA_CPP_STORAGE.current_config["chat_handler"] in ["Qwen3.5", "Qwen3.5-Thinking", "Qwen3.6", "Qwen3.6-Thinking"]:
                LLAMA_CPP_STORAGE.llm.n_tokens = 0
                LLAMA_CPP_STORAGE.llm._ctx.memory_clear(True)
                if LLAMA_CPP_STORAGE.llm.is_hybrid and LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr is not None:
                    LLAMA_CPP_STORAGE.llm._hybrid_cache_mgr.clear()

        del messages
        gc.collect()
        return (out1, out2, uid)


class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_tokens": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 1}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 1000, "step": 1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "typical_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "present_penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                #"tfs_z": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                #"penalty_last_n": ("INT", {"default": 64, "min": -1, "max": 8192, "step": 1}),
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
