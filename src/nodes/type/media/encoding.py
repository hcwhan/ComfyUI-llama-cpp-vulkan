"""媒体编码工具, ComfyUI 张量/音频 dict 转 base64, 供 image/video/audio Instruct 注入消息."""

import base64
import io
import wave

import numpy as np
import torch
from PIL import Image

from ....shared.logger import logger


def tensor_to_uint8(image: torch.Tensor):
    """ComfyUI IMAGE 张量 ([H,W,C] 或 [1,H,W,C]) 转 uint8 数组.

    只剥离批次维, 不用 squeeze(): squeeze 会把 H=1/W=1 的边缘尺寸也压掉,
    导致 PIL 把 [W,C] 误解析为灰度图.
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
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
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
    """ComfyUI AUDIO dict ({"waveform": [B,C,T], "sample_rate": int}) 转 WAV base64.

    只负责打包成 16-bit PCM WAV;重采样和声道适配由 llama.cpp 的 mtmd
    解码端完成,多声道先均值混为单声道以减小 base64 载荷.
    """
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        if waveform.shape[0] > 1:
            logger.warning(f"[llama-cpp-vulkan] AUDIO batch of {waveform.shape[0]} clips received; only the first clip is processed")
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    samples = np.clip(waveform.cpu().numpy(), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    return _encode_wav_base64(pcm.tobytes(), int(audio["sample_rate"]))


def scale_image(image: torch.Tensor, max_size: int):
    arr = tensor_to_uint8(image)
    h, w = arr.shape[:2]
    if max(w, h) <= max_size:
        # 不超上限时跳过等尺寸 LANCZOS 重采样(视频多帧场景逐帧调用, 白耗 CPU)
        return arr

    scale = max_size / max(w, h)
    # 极端长宽比下缩放结果可能取整为 0，至少保留 1 像素
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)

    return np.array(img_resized)


def image_content_item(uint8_image):
    """uint8 数组 -> chat 消息里的 image_url 内容项."""
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image2base64(uint8_image)}"}}
