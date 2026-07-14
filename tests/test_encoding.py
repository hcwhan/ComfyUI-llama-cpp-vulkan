"""src/nodes/type/media/encoding.py 的单元测试: 张量转 uint8, 音频打包, 图片缩放."""

import base64
import io
import unittest
import wave

import numpy as np
import torch

from tests import comfy_stubs

comfy_stubs.install()

from src.nodes.type.media.encoding import audio2base64, scale_image, tensor_to_uint8  # noqa: E402


def _decode_wav(b64):
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), samples


class TestTensorToUint8(unittest.TestCase):
    def test_edge_sizes_not_squeezed(self):
        # 回归: H=1/W=1 的边缘尺寸不得被压掉 (PIL 会把 [W,C] 误解析为灰度图)
        self.assertEqual(tensor_to_uint8(torch.zeros((1, 1, 5, 3))).shape, (1, 5, 3))
        self.assertEqual(tensor_to_uint8(torch.zeros((1, 5, 1, 3))).shape, (5, 1, 3))

    def test_3d_input_kept_as_is(self):
        self.assertEqual(tensor_to_uint8(torch.zeros((4, 6, 3))).shape, (4, 6, 3))

    def test_values_scaled_and_clipped(self):
        t = torch.tensor([[[0.0, 0.5, 1.0], [-1.0, 2.0, 0.25]]])
        arr = tensor_to_uint8(t)
        self.assertEqual(arr.tolist(), [[[0, 127, 255], [0, 255, 63]]])


class TestAudio2Base64(unittest.TestCase):
    def test_multichannel_mean_mix_to_mono(self):
        # 双声道取均值混为单声道: +0.5 与 -0.5 混合后为静音
        waveform = torch.stack([torch.full((8,), 0.5), torch.full((8,), -0.5)]).unsqueeze(0)
        channels, width, rate, samples = _decode_wav(audio2base64({"waveform": waveform, "sample_rate": 16000}))
        self.assertEqual((channels, width, rate), (1, 2, 16000))
        self.assertTrue((samples == 0).all())

    def test_out_of_range_samples_clipped(self):
        waveform = torch.tensor([[[2.0, -2.0, 1.0]]])
        _, _, _, samples = _decode_wav(audio2base64({"waveform": waveform, "sample_rate": 8000}))
        self.assertEqual(samples.tolist(), [32767, -32767, 32767])

    def test_batch_takes_first_clip(self):
        clip0 = torch.full((1, 4), 0.5)
        clip1 = torch.zeros((1, 4))
        waveform = torch.stack([clip0, clip1])
        _, _, _, samples = _decode_wav(audio2base64({"waveform": waveform, "sample_rate": 8000}))
        self.assertTrue((samples == 16383).all())


class TestScaleImage(unittest.TestCase):
    def test_extreme_aspect_keeps_min_1px(self):
        # 1000x1 的极端长宽比缩到 128: 短边取整为 0 时至少保留 1 像素
        arr = scale_image(torch.zeros((1000, 1, 3)), 128)
        self.assertEqual(arr.shape, (128, 1, 3))

    def test_oversized_image_scaled_to_max_size(self):
        arr = scale_image(torch.zeros((200, 400, 3)), 100)
        self.assertEqual(arr.shape, (50, 100, 3))

    def test_image_within_limit_returned_unchanged(self):
        # 回归: 不超 max_size 时短路返回, 跳过等尺寸重采样, 像素值不得有重采样扰动
        image = torch.rand((32, 64, 3))
        arr = scale_image(image, 128)
        self.assertEqual(arr.shape, (32, 64, 3))
        self.assertTrue((arr == tensor_to_uint8(image)).all())


if __name__ == "__main__":
    unittest.main()
