// 插件节点的默认尺寸 (纯 UX 增强): 新建节点时按节点组设置默认宽度
// (loader 与 Instruct 420, 透传/拆分类小节点 240, 其余 300), 并给提示词
// 输入框补默认显示高度 (见 widget_utils.js 的 promptHeightBump).
// 只覆盖创建时刻的默认值: computeSize 不动, 手动拖拽仍可缩到自然下限;
// 载入工作流时 nodeCreated 先于 configure 的尺寸恢复执行, 保存的尺寸
// 原样生效. 与 vlm_loader.js / image_instruct.js 的首次 SNAP 重排
// (宽度只增不减, 高度同含补高) 在任意扩展执行顺序下收敛到同一结果.
import { app } from "../../scripts/app.js";
import { promptHeightBump } from "./widget_utils.js";

const CATEGORY = "llama-cpp-vulkan";

// loader 与 Instruct 节点字段多 (Instruct 另含多行提示词输入框), 默认更宽
const WIDE_NODES = new Set([
    "llama_cpp_llm_model_loader",
    "llama_cpp_vlm_model_loader",
    "llama_cpp_text_instruct",
    "llama_cpp_image_instruct",
    "llama_cpp_video_instruct",
    "llama_cpp_audio_instruct",
]);
// 透传/拆分类小节点字段少, 默认更窄
const NARROW_NODES = new Set([
    "llama_cpp_unload_model",
    "remove_code_block",
    "split_instruct_output",
]);
const WIDE_WIDTH = 420;
const NARROW_WIDTH = 240;
const DEFAULT_WIDTH = 300;

const pluginNodes = new Set();

app.registerExtension({
    name: "llama-cpp-vulkan.node-size",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.category === CATEGORY) {
            pluginNodes.add(nodeData.name);
        }
    },
    nodeCreated(node) {
        if (!pluginNodes.has(node.comfyClass)) {
            return;
        }
        const computed = node.computeSize();
        let width = DEFAULT_WIDTH;
        if (WIDE_NODES.has(node.comfyClass)) {
            width = WIDE_WIDTH;
        } else if (NARROW_NODES.has(node.comfyClass)) {
            width = NARROW_WIDTH;
        }
        // 内容计算宽超过组默认宽时以计算宽为准, 不裁内容
        node.setSize([Math.max(computed[0], width), computed[1] + promptHeightBump(node)]);
    },
});
