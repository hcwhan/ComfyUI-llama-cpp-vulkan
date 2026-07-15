// image Instruct 的 widget 联动 (纯 UX 增强, 行为正确性由 Python 侧保证):
// - max_size 仅在 mode 为 Batch 时显示 (Per-Image 逐张推理不缩放, 该字段无作用),
//   隐藏字段的值仍序列化并随工作流保存, 重新显示时保留原值
// Batch 档的选项值来自 mode widget options 的自定义 key batch_mode_value
// (/object_info 原样透传, 与 common_static.py 的常量单一真源).
import { app } from "../../scripts/app.js";
import { toggleWidget } from "./widget_utils.js";

const NODE_NAME = "llama_cpp_image_instruct";

let batchModeValue = null;

app.registerExtension({
    name: "llama-cpp-vulkan.image-instruct",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }
        batchModeValue = nodeData.input?.required?.mode?.[1]?.batch_mode_value ?? null;
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) {
            return;
        }
        const modeWidget = node.widgets?.find((w) => w.name === "mode");
        const maxSizeWidget = node.widgets?.find((w) => w.name === "max_size");
        if (!modeWidget || !maxSizeWidget || batchModeValue === null) {
            return;
        }

        const applyMode = () => {
            // 仅在显隐实际变化时重排节点尺寸, 避免覆盖用户手动调整的大小
            if (toggleWidget(maxSizeWidget, modeWidget.value === batchModeValue)) {
                node.setSize(node.computeSize());
            }
        };

        const originalCallback = modeWidget.callback;
        modeWidget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            applyMode();
            return result;
        };

        // 新建节点立即同步; 载入工作流时 configure 恢复 widget 值后再同步
        applyMode();
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = originalOnConfigure?.apply(this, args);
            applyMode();
            return result;
        };
    },
});
