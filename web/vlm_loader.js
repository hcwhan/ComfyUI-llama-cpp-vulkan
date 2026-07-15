// vlm Model Loader 的 widget 联动 (纯 UX 增强, 行为正确性由 Python 侧保证):
// - thinking 开关按 chat_handler 选值三态置灰 (可切换 / 强制开 / 强制关),
//   强制档覆写前缓存用户在可切换档的设定, 切回可切换档时恢复;
//   失效时由 clamp_thinking 钳制兜底
// - image_min/max_tokens 仅在选中支持视觉的 handler 时显示 (音频专用与
//   "None" 隐藏), 隐藏字段的值仍序列化并随 config 下发, 对音频路径无效
// - image_min/max_tokens 区间互钳: 修改任一侧越界时钳制另一侧 (max=0 视为
//   未设置不参与), 前端阻止区间倒挂; loader 侧报错保留, 兜底绕过 UI 的提交
// 能力名单来自 chat_handler widget options 的自定义 key thinking_modes /
// image_token_handlers (/object_info 原样返回), 与 Python 注册表单一真源.
// 联动只原位改值/置灰/改类型, 不增删 widget (widgets_values 按声明序序列化).
import { app } from "../../scripts/app.js";
import { toggleWidget } from "./widget_utils.js";

const NODE_NAME = "llama_cpp_vlm_model_loader";

let thinkingModes = null;
let imageTokenHandlers = null;

app.registerExtension({
    name: "llama-cpp-vulkan.vlm-loader",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) {
            return;
        }
        const options = nodeData.input?.required?.chat_handler?.[1] ?? {};
        thinkingModes = options.thinking_modes ?? null;
        imageTokenHandlers = options.image_token_handlers ? new Set(options.image_token_handlers) : null;
    },
    nodeCreated(node) {
        if (node.comfyClass !== NODE_NAME) {
            return;
        }
        const handlerWidget = node.widgets?.find((w) => w.name === "chat_handler");
        if (!handlerWidget) {
            return;
        }
        const thinkingWidget = node.widgets.find((w) => w.name === "thinking");
        const imageTokenWidgets = node.widgets.filter((w) => w.name === "image_min_tokens" || w.name === "image_max_tokens");

        // forced/none 档覆写 thinking 值前, 把用户在 toggle 档的设定缓存到
        // _userValue (仅在从 toggle 档切出的那一次记录), 切回 toggle 档时恢复;
        // prevThinkingMode 起始为 null, 使 nodeCreated/onConfigure 的首次同步
        // 不产生缓存, 载入工作流以序列化值为准
        let prevThinkingMode = null;
        const applyMode = () => {
            const label = handlerWidget.value;
            if (thinkingWidget && thinkingModes) {
                // "None" 与未知 label 按不支持处理 (无 handler 即无思考模式)
                const mode = thinkingModes[label] ?? "none";
                if (mode === "toggle") {
                    if (thinkingWidget._userValue !== undefined) {
                        thinkingWidget.value = thinkingWidget._userValue;
                        delete thinkingWidget._userValue;
                    }
                    thinkingWidget.disabled = false;
                } else {
                    if (prevThinkingMode === "toggle") {
                        thinkingWidget._userValue = thinkingWidget.value;
                    }
                    thinkingWidget.value = mode === "forced";
                    thinkingWidget.disabled = true;
                }
                prevThinkingMode = mode;
            }
            if (imageTokenHandlers) {
                const show = imageTokenHandlers.has(label);
                let changed = false;
                for (const widget of imageTokenWidgets) {
                    changed = toggleWidget(widget, show) || changed;
                }
                // 仅在显隐实际变化时重排节点尺寸, 避免覆盖用户手动调整的大小
                if (changed) {
                    node.setSize(node.computeSize());
                }
            }
        };

        const originalCallback = handlerWidget.callback;
        handlerWidget.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            applyMode();
            return result;
        };

        // 区间互钳: 只在用户修改一侧时动另一侧, 载入工作流不改存量值
        // (倒挂的存量值由 loader 侧报错兜底)
        const minWidget = node.widgets.find((w) => w.name === "image_min_tokens");
        const maxWidget = node.widgets.find((w) => w.name === "image_max_tokens");
        if (minWidget && maxWidget) {
            const hookClamp = (widget, clamp) => {
                const original = widget.callback;
                widget.callback = function (...args) {
                    const result = original?.apply(this, args);
                    clamp();
                    return result;
                };
            };
            hookClamp(minWidget, () => {
                if (maxWidget.value > 0 && minWidget.value > maxWidget.value) {
                    maxWidget.value = minWidget.value;
                }
            });
            hookClamp(maxWidget, () => {
                if (maxWidget.value > 0 && maxWidget.value < minWidget.value) {
                    minWidget.value = maxWidget.value;
                }
            });
        }

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
