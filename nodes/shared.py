import os
import io
import re
import json
import wave
import base64
import hashlib
import torch

import numpy as np
from PIL import Image, ImageDraw

import folder_paths

llm_extensions = {'.gguf'}
_LLM_FOLDER_KEYS = ("llm", "LLM")
for _key in _LLM_FOLDER_KEYS:
    folder_paths.add_model_folder_path(_key, os.path.join(folder_paths.models_dir, _key))
    folder_paths.folder_names_and_paths[_key][1].update(llm_extensions)


def get_llm_filename_list():
    seen = set()
    result = []
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            for f in folder_paths.get_filename_list(key):
                if f not in seen:
                    seen.add(f)
                    result.append(f)
    return result


def get_llm_full_path(filename):
    for key in _LLM_FOLDER_KEYS:
        if key in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(key, filename)
            if path is not None:
                return path
    return None


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

preset_prompts = {
    "Empty - Nothing": "",
    "Normal - Describe": "Describe this @.",
    "Prompt Style - Tags": "Your task is to generate a clean list of comma-separated tags for a text-to-@ AI, based *only* on the visual information in the @. Limit the output to a maximum of 50 unique tags. Strictly describe visual elements like subject, clothing, environment, colors, lighting, and composition. Do not include abstract concepts, interpretations, marketing terms, or technical jargon (e.g., no 'SEO', 'brand-aligned', 'viral potential'). The goal is a concise list of visual descriptors. Avoid repeating tags.",
    "Prompt Style - Simple": "Analyze the @ and generate a simple, single-sentence text-to-@ prompt. Describe the main subject and the setting concisely.",
    "Prompt Style - Detailed": "Generate a detailed, artistic text-to-@ prompt based on the @. Combine the subject, their actions, the environment, lighting, and overall mood into a single, cohesive paragraph of about 2-3 sentences. Focus on key visual details.",
    "Prompt Style - Extreme Detailed": "Generate an extremely detailed and descriptive text-to-@ prompt from the @. Create a rich paragraph that elaborates on the subject's appearance, textures of clothing, specific background elements, the quality and color of light, shadows, and the overall atmosphere. Aim for a highly descriptive and immersive prompt.",
    "Prompt Style - Cinematic": "Act as a master prompt engineer. Create a highly detailed and evocative prompt for an @ generation AI. Describe the subject, their pose, the environment, the lighting, the mood, and the artistic style (e.g., photorealistic, cinematic, painterly). Weave all elements into a single, natural language paragraph, focusing on visual impact.",
    "Creative - Detailed Analysis": "Describe this @ in detail, breaking down the subject, attire, accessories, background, and composition into separate sections.",
    "Creative - Summarize Video": "Summarize the key events and narrative points in this video.",
    "Creative - Short Story": "Write a short, imaginative story inspired by this @ or video.",
    "Creative - Refine & Expand Prompt": "Refine and enhance the following user prompt for creative text-to-@ generation. Keep the meaning and keywords, make it more expressive and visually rich. Output **only the improved prompt text itself**, without any reasoning steps, thinking process, or additional commentary.",
    "Vision - *Bounding Box": 'Locate every instance that belongs to the following categories: "#". Report bbox coordinates in {"bbox_2d": [x1, y1, x2, y2], "label": "string"} JSON format as a List.'
}
preset_tags = list(preset_prompts.keys())


def tensor_to_uint8(image: torch.Tensor):
    """ComfyUI IMAGE 张量 ([H,W,C] 或 [1,H,W,C]) 转 uint8 数组。

    只剥离 batch 维，不用 squeeze()：squeeze 会把 H=1/W=1 的边缘尺寸也压掉，
    导致 PIL 把 [W,C] 误解析为灰度图。
    """
    arr = image.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return np.clip(255.0 * arr, 0, 255).astype(np.uint8)


def image2base64(image):
    # PNG 无损：JPEG q85 的压缩伪影会影响 OCR 类模型的小字识别，
    # 且 JPEG 不支持 RGBA 输入
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64


def _encode_wav_base64(pcm_bytes, sample_rate):
    buffered = io.BytesIO()
    with wave.open(buffered, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def audio2base64(audio):
    """ComfyUI AUDIO dict ({"waveform": [B,C,T], "sample_rate": int}) 转 WAV base64。

    只负责打包成 16-bit PCM WAV;重采样和声道适配由 llama.cpp 的 mtmd
    解码端完成,多声道先均值混为单声道以减小 base64 载荷。
    """
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    samples = np.clip(waveform.cpu().numpy(), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    return _encode_wav_base64(pcm.tobytes(), int(audio["sample_rate"]))


# 开头的 ```label(标签限单词类字符,可无,如 json/python/c++);结尾的 ```。
# 两端独立匹配,生成被截断导致围栏未闭合时,开头的标记仍能剥离。
# 标签不能用 [^\s`]* 之类的宽匹配:围栏后无换行直接跟正文时会把正文吞掉
_FENCE_OPEN_RE = re.compile(r"^```[\w+.-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```$")


def strip_code_fence(text, label=""):
    """去除 LLM 输出首尾的 ```label ... ``` 代码块标记。

    label 仅为语义提示,实际兼容任意标签和裸 ``` 围栏:
    模型即使被要求输出 json 也可能给出不带标签的围栏。
    """
    text = text.strip()
    text = _FENCE_OPEN_RE.sub("", text)
    return _FENCE_CLOSE_RE.sub("", text)


def parse_json(json_str):
    try:
        parsed = json.loads(strip_code_fence(json_str, "json"))
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed


def scale_image(image: torch.Tensor, max_size: int = 128):
    img_pil = Image.fromarray(tensor_to_uint8(image))

    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    # 极端长宽比下缩放结果可能取整为 0，至少保留 1 像素
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return np.array(img_resized)


QWEN_BBOX_MODES = ("Qwen3-VL", "Qwen2.5-VL")


def bbox_label(item):
    """取 bbox JSON 项的标签,兼容 label / text_content 两种字段。"""
    return item.get("label") or item.get("text_content") or "bbox"


def json_to_pixel_bboxes(json_items, mode, width=0, height=0):
    """把 LLM 输出的 bbox JSON 项换算为像素坐标 [(x0, y0, x1, y1), ...]。

    Qwen 系列模型输出 0-1000 归一化坐标,需按图像尺寸换算;
    simple 模式视为已是像素坐标,原样透传。
    """
    bboxes = []
    for item in json_items:
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in QWEN_BBOX_MODES:
            x0 = x0 / 1000 * width
            y0 = y0 / 1000 * height
            x1 = x1 / 1000 * width
            y1 = y1 / 1000 * height
        bboxes.append((x0, y0, x1, y1))
    return bboxes


def _label_color(label):
    # 由 label 内容哈希出稳定颜色,同一 label 每次运行颜色一致;
    # 80-180 区间保证中等亮度,白色标签文字可读
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return tuple(80 + b % 101 for b in digest[:3])


def draw_bbox(image, pixel_bboxes, labels):
    img = Image.fromarray(tensor_to_uint8(image))
    draw = ImageDraw.Draw(img)

    for (x0, y0, x1, y1), label in zip(pixel_bboxes, labels):
        color = _label_color(label)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)


def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            # 嵌套的 JSON-in-string 字段:解析失败视为无法下钻,回落 default,
            # 与 "key 不存在" 的语义一致(顶层输入的解析错误由 parse_json 报出)
            try:
                data = json.loads(data)
            except ValueError:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data
