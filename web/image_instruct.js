// image Instruct 的 widget 联动 (纯 UX 增强, 行为正确性由 Python 侧保证):
// - increment_seed 仅在 mode 为 Per-Image 时显示 (Batch 单次请求无逐张派生语义),
// - max_size 仅在 mode 为 Batch 时显示 (Per-Image 逐张推理不缩放, 该字段无作用),
//   隐藏字段的值仍序列化并随工作流保存, 重新显示时保留原值
// 两档的选项值来自 mode widget options 的自定义 key each_mode_value /
// batch_mode_value (/object_info 原样透传, 与 common_static.py 的常量单一真源),
// 每个 widget 各自与对应档位值显式比对 (枚举分支规范).
import { app } from "../../scripts/app.js";
import { REFLOW_GROW, REFLOW_NONE, REFLOW_SNAP, reflowNode, toggleWidget } from "./widget_utils.js";

const NODE_NAME = "llama_cpp_image_instruct";

let eachModeValue = null;
let batchModeValue = null;

app.registerExtension({
    name: "llama-cpp-vulkan.image-instruct",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }
        const options = nodeData.input?.required?.mode?.[1] ?? {};
        eachModeValue = options.each_mode_value ?? null;
        batchModeValue = options.batch_mode_value ?? null;
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) {
            return;
        }
        const modeWidget = node.widgets?.find((w) => w.name === "mode");
        const incrementSeedWidget = node.widgets?.find((w) => w.name === "increment_seed");
        const maxSizeWidget = node.widgets?.find((w) => w.name === "max_size");
        if (!modeWidget || !incrementSeedWidget || !maxSizeWidget || eachModeValue === null || batchModeValue === null) {
            return;
        }

        const applyMode = (reflowMode) => {
            // 两个 toggle 都要执行 (勿短路), 仅在显隐实际变化时按调用路径的
            // 模式重排 (各模式语义见 widget_utils.js): 交互切档只增不减,
            // 载入路径不重排
            const incrementSeedChanged = toggleWidget(incrementSeedWidget, modeWidget.value === eachModeValue);
            const maxSizeChanged = toggleWidget(maxSizeWidget, modeWidget.value === batchModeValue);
            if (incrementSeedChanged || maxSizeChanged) {
                reflowNode(node, reflowMode);
            }
        };

        const originalCallback = modeWidget.callback;
        modeWidget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            applyMode(REFLOW_GROW);
            return result;
        };

        // 新建节点立即同步 (SNAP 收紧隐藏字段占用的高度, 宽度与
        // node_size.js 的默认宽度设置收敛; 载入路径此次也会执行, 但发生在
        // configure 恢复尺寸之前, 无害); 载入工作流时 configure 恢复
        // widget 值后再同步, 此时序列化尺寸已恢复, 不重排以免覆盖
        applyMode(REFLOW_SNAP);
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = originalOnConfigure?.apply(this, args);
            applyMode(REFLOW_NONE);
            return result;
        };
    },
});
