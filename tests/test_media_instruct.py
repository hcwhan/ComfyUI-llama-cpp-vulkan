"""video / audio Instruct process() 全链路的节点级单元测试.

用 FakeVlm 替身走通 process() (不加载真实模型), 锁定胶水层:
- video: "连续视频" 语义提示注入 system prompt (空 system_prompt 时单独
  成条, 非空时前置), 均匀抽帧 -> 多帧逐帧缩放到 max_size (单帧保持原
  分辨率) -> 全部帧并入单条 user 消息
- video: 抽帧接线以可区分像素锁定 (选中的是均匀间隔帧而非前 N 帧)
- audio: AUDIO dict 经 audio2base64 混为单声道 16-bit WAV,
  以 input_audio 内容项注入单条 user 消息
- video/audio: require_mmproj 失败分支 (chat_handler 无 mmproj_path 时
  抛 ValueError, 不发起请求)
"""

import base64
import io
import re
import types
import unittest
import wave
from unittest import mock

import torch
from PIL import Image

from tests import comfy_stubs

comfy_stubs.install()

from src.core.storage import LLAMA_CPP_STORAGE  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402
from src.nodes.instruct.media.audio.node_instruct import llama_cpp_audio_instruct  # noqa: E402
from src.nodes.instruct.media.video.node_instruct import llama_cpp_video_instruct  # noqa: E402


class _FakeVlm:
    """process() 全链路所需的最小 llm 替身: 返回预置文本并记录消息, 附 _run 收尾判定属性."""

    def __init__(self, output):
        self._output = output
        self.calls = 0
        self.last_messages = None
        self.n_tokens = 0
        self._model = types.SimpleNamespace(is_hybrid=lambda: False, is_recurrent=lambda: False)
        self._ctx = mock.Mock()
        self._hybrid_cache_mgr = None

    def abort(self):
        pass

    def create_chat_completion(self, messages, seed, **params):
        self.last_messages = messages
        self.calls += 1
        return {
            "usage": {"prompt_tokens": 20, "completion_tokens": 100},
            "choices": [{"message": {"content": self._output}}],
        }


def _png_size(content_item):
    """image_url 内容项的 data URL -> PNG 实际 (宽, 高), 用于断言缩放行为."""
    b64 = content_item["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def _png_pixel(content_item):
    """image_url 内容项的 data URL -> PNG (0, 0) 像素 R 通道值, 用于区分选中的是哪一帧."""
    b64 = content_item["image_url"]["url"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).getpixel((0, 0))[0]


class _MediaInstructTestBase(unittest.TestCase):
    def setUp(self):
        self.config = {"model": "m.gguf"}
        self.llm = _FakeVlm("结果文本")
        self._orig_state = (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config)
        self.addCleanup(self._restore_state)
        # current_config 与 llama_model 相同, _prepare_messages 不触发真实加载;
        # chat_handler 带 mmproj_path 满足 require_mmproj 校验
        LLAMA_CPP_STORAGE.llm = self.llm
        LLAMA_CPP_STORAGE.chat_handler = types.SimpleNamespace(mmproj_path="fake.mmproj")
        LLAMA_CPP_STORAGE.current_config = self.config

    def _restore_state(self):
        (LLAMA_CPP_STORAGE.llm, LLAMA_CPP_STORAGE.chat_handler, LLAMA_CPP_STORAGE.current_config) = self._orig_state

    def _user_content(self):
        (user_msg,) = [m for m in self.llm.last_messages if m["role"] == "user"]
        return user_msg["content"]


class TestVideoInstructProcess(_MediaInstructTestBase):
    def _process(self, frames, system_prompt="", max_frames=30, max_size=256):
        (out,) = llama_cpp_video_instruct().process(
            vlm_model=self.config,
            frames=frames,
            seed=0,
            preset_prompt="空白 - 空",
            custom_prompt="描述这段视频",
            system_prompt=system_prompt,
            max_frames=max_frames,
            max_size=max_size,
            strip_thinking=True,
            force_offload=False,
        )
        return out

    def test_multi_frame_sampled_scaled_into_single_message(self):
        # 5 帧灌入可区分像素 (i/4 -> uint8 0/63/127/191/255), 按 max_frames=3
        # 均匀抽帧应选中第 0/2/4 帧 (解码像素 0/127/255; 若取帧行回归为
        # frames[:3], 像素序列变为 0/63/127, 本断言即报出); 多帧逐帧缩放
        # (H8xW16 按 max_size=8 -> H4xW8, 断言取 PIL (宽, 高) 即 (8, 4),
        # 纯色帧缩放不改变像素值), 全部帧并入单条 user 消息
        # (文本项 + 3 个 image_url 项), 一次推理
        frames = torch.stack([torch.full((8, 16, 3), i / 4.0) for i in range(5)])
        out = self._process(frames, max_frames=3, max_size=8)
        self.assertEqual(out, "结果文本")
        self.assertEqual(self.llm.calls, 1)
        content = self._user_content()
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url", "image_url"])
        self.assertEqual(content[0]["text"], "描述这段视频")
        for item in content[1:]:
            self.assertEqual(_png_size(item), (8, 4))
        self.assertEqual([_png_pixel(item) for item in content[1:]], [0, 127, 255])

    def test_single_frame_keeps_original_resolution(self):
        # 单帧不缩放: 16 宽超出 max_size=8 也保持原分辨率
        self._process(torch.zeros(1, 8, 16, 3), max_size=8)
        content = self._user_content()
        self.assertEqual([item["type"] for item in content], ["text", "image_url"])
        self.assertEqual(_png_size(content[1]), (16, 8))

    def test_video_hint_injected_as_system_message(self):
        # 空 system_prompt 时 "连续视频" 语义提示单独成条 system 消息
        self._process(torch.zeros(2, 4, 4, 3))
        system = self.llm.last_messages[0]
        self.assertEqual(system["role"], "system")
        self.assertIn("连续的视频", system["content"])

    def test_video_hint_prepended_to_user_system_prompt(self):
        # 非空 system_prompt 时语义提示前置, 用户内容保留在其后
        self._process(torch.zeros(2, 4, 4, 3), system_prompt="你是视频分析助手")
        system = self.llm.last_messages[0]
        self.assertEqual(system["role"], "system")
        self.assertIn("连续的视频", system["content"])
        self.assertTrue(system["content"].endswith("你是视频分析助手"))

    def test_missing_mmproj_raises_before_request(self):
        # runner 第一句 require_mmproj 校验: chat_handler 无 mmproj_path 时
        # 抛 ValueError (文案引用 LANG), 且不发起 completion 请求
        LLAMA_CPP_STORAGE.chat_handler = types.SimpleNamespace(mmproj_path=None)
        expected = LANG["nodes"]["instruct"]["common"]["errors"]["mmproj_not_configured"].format(kind="Video")
        with self.assertRaisesRegex(ValueError, re.escape(expected)):
            self._process(torch.zeros(2, 4, 4, 3))
        self.assertEqual(self.llm.calls, 0)


class TestAudioInstructProcess(_MediaInstructTestBase):
    def _process(self, audio):
        (out,) = llama_cpp_audio_instruct().process(
            vlm_model=self.config,
            audio=audio,
            seed=0,
            preset_prompt="空白 - 空",
            custom_prompt="转写这段音频",
            system_prompt="",
            strip_thinking=True,
            force_offload=False,
        )
        return out

    def test_audio_injected_as_input_audio_wav(self):
        # AUDIO dict ([B,C,T] 双声道) 混为单声道 16-bit WAV,
        # 以 input_audio 内容项并入单条 user 消息
        audio = {"waveform": torch.zeros(1, 2, 100), "sample_rate": 16000}
        out = self._process(audio)
        self.assertEqual(out, "结果文本")
        self.assertEqual(self.llm.calls, 1)
        content = self._user_content()
        self.assertEqual([item["type"] for item in content], ["text", "input_audio"])
        self.assertEqual(content[0]["text"], "转写这段音频")
        self.assertEqual(content[1]["input_audio"]["format"], "wav")
        with wave.open(io.BytesIO(base64.b64decode(content[1]["input_audio"]["data"])), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnframes(), 100)

    def test_missing_mmproj_raises_before_request(self):
        # runner 第一句 require_mmproj 校验: chat_handler 无 mmproj_path 时
        # 抛 ValueError (文案引用 LANG), 且不发起 completion 请求
        LLAMA_CPP_STORAGE.chat_handler = types.SimpleNamespace(mmproj_path=None)
        expected = LANG["nodes"]["instruct"]["common"]["errors"]["mmproj_not_configured"].format(kind="Audio")
        with self.assertRaisesRegex(ValueError, re.escape(expected)):
            self._process({"waveform": torch.zeros(1, 2, 100), "sample_rate": 16000})
        self.assertEqual(self.llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
