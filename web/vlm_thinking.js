// vlm Model Loader 的 thinking 开关三态联动: 按 chat_handler 选值把开关
// 置为 可切换 / 强制开置灰 / 强制关置灰. 能力名单来自 chat_handler widget
// options 的自定义 key thinking_modes (/object_info 原样返回), 与 Python
// 注册表单一真源; 本扩展失效只损失置灰效果, 行为正确性由 Python 侧
// clamp_thinking 钳制兜底. 三态切换只原位改值/置灰, 不增删 widget
// (widgets_values 按声明序序列化).
import { app } from "../../scripts/app.js";

const NODE_NAME = "llama_cpp_vlm_model_loader";

let thinkingModes = null;

app.registerExtension({
    name: "llama-cpp-vulkan.vlm-thinking",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === NODE_NAME) {
            thinkingModes = nodeData.input?.required?.chat_handler?.[1]?.thinking_modes ?? null;
        }
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME || !thinkingModes) {
            return;
        }
        const handlerWidget = node.widgets?.find((w) => w.name === "chat_handler");
        const thinkingWidget = node.widgets?.find((w) => w.name === "thinking");
        if (!handlerWidget || !thinkingWidget) {
            return;
        }

        const applyMode = () => {
            // "None" 与未知 label 按不支持处理 (无 handler 即无思考模式)
            const mode = thinkingModes[handlerWidget.value] ?? "none";
            if (mode === "forced") {
                thinkingWidget.value = true;
                thinkingWidget.disabled = true;
            } else if (mode === "none") {
                thinkingWidget.value = false;
                thinkingWidget.disabled = true;
            } else {
                thinkingWidget.disabled = false;
            }
        };

        const originalCallback = handlerWidget.callback;
        handlerWidget.callback = function (...args) {
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
