"""src/core/devices.py 的单元测试: 显式选卡翻译与回退, log_backend_summary 文案分支."""

import unittest

from tests import comfy_stubs

comfy_stubs.install()

from src.core import devices  # noqa: E402
from src.i18n.lang import LANG  # noqa: E402

_FAKE_DGPU = {"name": "Vulkan0", "desc": "Fake dGPU", "type": "GPU", "device_id": "0000:03:00.0"}
_FAKE_IGPU = {"name": "Vulkan1", "desc": "Fake iGPU", "type": "IGPU", "device_id": None}
_FAKE_DGPU2 = {"name": "Vulkan2", "desc": "Fake dGPU 2", "type": "GPU", "device_id": "0000:04:00.0"}


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


class TestSelectableDeviceDedup(unittest.TestCase):
    def setUp(self):
        self._orig = devices._gpu_devices
        self.addCleanup(setattr, devices, "_gpu_devices", self._orig)

    def test_same_device_id_keeps_first_enumerated(self):
        # 回归: 修复前未复刻上游 same-device_id 去重 (llama.cpp 收集
        # model->devices 时跳过与已收集独显同 id 者), 同一物理卡被双 ICD
        # 枚举两次时下拉框会列出两项, 索引与 llama.cpp 设备列表错位
        first = {"name": "Vulkan0", "desc": "Card via ICD A", "type": "GPU", "device_id": "0000:03:00.0"}
        second = {"name": "Vulkan1", "desc": "Card via ICD B", "type": "GPU", "device_id": "0000:03:00.0"}
        devices._gpu_devices = [first, second]
        self.assertEqual(devices._selectable_devices(), [first])

    def test_distinct_device_ids_all_kept(self):
        devices._gpu_devices = [_FAKE_DGPU, _FAKE_DGPU2]
        self.assertEqual(devices._selectable_devices(), [_FAKE_DGPU, _FAKE_DGPU2])

    def test_unknown_device_id_never_deduped(self):
        # 上游仅在双方 device_id 均非空时比较, id 未知 (C 侧 NULL, 插件映射
        # 为 None) 的设备一律保留
        first = {"name": "Vulkan0", "desc": "Fake dGPU", "type": "GPU", "device_id": None}
        second = {"name": "Vulkan1", "desc": "Fake dGPU 2", "type": "GPU", "device_id": None}
        devices._gpu_devices = [first, second]
        self.assertEqual(devices._selectable_devices(), [first, second])


class TestLogBackendSummary(unittest.TestCase):
    def setUp(self):
        self._orig = devices._gpu_devices
        self.addCleanup(setattr, devices, "_gpu_devices", self._orig)

    def test_multi_gpu_layer_split_lists_all_names(self):
        devices._gpu_devices = [_FAKE_DGPU, _FAKE_DGPU2]
        with self.assertLogs("llama-cpp-vulkan", level="INFO") as logs:
            devices.log_backend_summary(0, devices.SPLIT_MODE_LAYER)
        expected = LANG["logs"]["devices"]["active_gpus_layer_split"].format(names="Vulkan0, Vulkan2")
        self.assertTrue(any(expected in m for m in logs.output))

    def test_explicit_selection_logs_single_device(self):
        # 显式选卡 (NONE 模式) 按 main_gpu 索引取可选列表中的设备
        devices._gpu_devices = [_FAKE_DGPU, _FAKE_DGPU2]
        with self.assertLogs("llama-cpp-vulkan", level="INFO") as logs:
            devices.log_backend_summary(1, devices.SPLIT_MODE_NONE)
        expected = LANG["logs"]["devices"]["active_gpu"].format(name="Vulkan2", desc="Fake dGPU 2", type="GPU")
        self.assertTrue(any(expected in m for m in logs.output))

    def test_no_backend_logs_warning(self):
        devices._gpu_devices = []
        with self.assertLogs("llama-cpp-vulkan", level="WARNING") as logs:
            devices.log_backend_summary(0, devices.SPLIT_MODE_LAYER)
        self.assertTrue(any(LANG["logs"]["devices"]["no_backend"] in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
