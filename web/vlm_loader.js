// VLM Model Loader 的 widget 联动 (纯 UX 增强, 行为正确性由 Python 侧保证):
// - thinking 开关按 chat_handler 选值三态置灰 (可切换 / 强制开 / 强制关),
//   强制档覆写前缓存可编辑状态的设定, 切回可编辑状态时恢复; 未知 label
//   (含 "None" 占位与 wheel 缺类) 不置灰不动值, 保存工作流不丢序列化设定;
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

        // forced/none 档覆写 thinking 值前, 把可编辑状态下的设定缓存到
        // _userValue (仅在从可编辑状态切出的那一次记录), 切回可编辑状态时恢复;
        // prevEditable 起始为 null, 使 nodeCreated/onConfigure 的首次同步
        // 不产生缓存, 载入工作流以序列化值为准
        let prevEditable = null;
        const applyMode = () => {
            const label = handlerWidget.value;
            if (thinkingWidget && thinkingModes) {
                // 未知 label (含 "None" 占位与 wheel 缺类) 与 toggle 档同属
                // 可编辑: 不置灰不覆写, 使缺类 handler 的工作流在保存时不丢
                // 序列化的 thinking 值; 实际生效值由 Python 侧钳制/校验兜底
                const mode = thinkingModes[label];
                const editable = mode !== "forced" && mode !== "none";
                if (editable) {
                    if (thinkingWidget._userValue !== undefined) {
                        thinkingWidget.value = thinkingWidget._userValue;
                        delete thinkingWidget._userValue;
                    }
                    thinkingWidget.disabled = false;
                } else {
                    if (prevEditable) {
                        thinkingWidget._userValue = thinkingWidget.value;
                    }
                    thinkingWidget.value = mode === "forced";
                    thinkingWidget.disabled = true;
                }
                prevEditable = editable;
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
