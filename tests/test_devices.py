"""src/core/devices.py resolve_device_selection 的单元测试: 显式选卡翻译与回退."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.core import devices  # noqa: E402

_FAKE_DGPU = {"name": "Vulkan0", "desc": "Fake dGPU", "type": "GPU"}
_FAKE_IGPU = {"name": "Vulkan1", "desc": "Fake iGPU", "type": "IGPU"}


class TestResolveDeviceSelection(unittest.TestCase):
    def setUp(self):
        # 模块级设备列表在 import 时经真实 Vulkan 枚举而来, 参数化后测翻译逻辑
        self._orig = devices._gpu_devices
        devices._gpu_devices = [_FAKE_IGPU, _FAKE_DGPU]
        self.addCleanup(setattr, devices, "_gpu_devices", self._orig)

    def test_auto_label_uses_layer_split(self):
        self.assertEqual(
            devices.resolve_device_selection(devices.AUTO_LABEL),
            (0, devices.SPLIT_MODE_LAYER),
        )

    def test_explicit_dgpu_uses_none_split(self):
        # 索引是可选列表 (只含独显) 中的位置, 不是 ggml 全局设备序号
        label = devices._device_label(_FAKE_DGPU)
        self.assertEqual(devices.resolve_device_selection(label), (0, devices.SPLIT_MODE_NONE))

    def test_igpu_label_falls_back_when_dgpu_present(self):
        # 有独显时核显不在可选列表, 显式传核显 label 回退 Auto
        label = devices._device_label(_FAKE_IGPU)
        self.assertEqual(devices.resolve_device_selection(label), (0, devices.SPLIT_MODE_LAYER))

    def test_unknown_label_falls_back_to_auto(self):
        self.assertEqual(
            devices.resolve_device_selection("no such device"),
            (0, devices.SPLIT_MODE_LAYER),
        )

    def test_igpu_selectable_when_no_dgpu(self):
        devices._gpu_devices = [_FAKE_IGPU]
        label = devices._device_label(_FAKE_IGPU)
        self.assertEqual(devices.resolve_device_selection(label), (0, devices.SPLIT_MODE_NONE))


if __name__ == "__main__":
    unittest.main()
