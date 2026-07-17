// web/node_size.js 的单元测试: category 过滤, 分组默认宽度 (loader 与
// Instruct 420, 透传/拆分类小节点 240, 其余 300), 内容计算宽更大时不裁,
// 提示词输入框补默认高度.

import assert from "node:assert/strict";
import { test } from "node:test";

import { extensions, loadWebModule, makeNode, makeWidget } from "./harness.mjs";

await loadWebModule("node_size.js");

const { PROMPT_WIDGET_DEFAULT_HEIGHT } = await import("../../web/widget_utils.js");

const ext = extensions.find((e) => e.name === "llama-cpp-vulkan.node-size");

// 模块级 pluginNodes 集合只增不减, 各测试注册互不同名的节点定义即可隔离
const registerDef = (name, category = "llama-cpp-vulkan") => {
    ext.beforeRegisterNodeDef(null, { name, category });
};

// 与前端 customtext DOM widget 的最小对齐: type + computeLayoutSize
const makePromptWidget = () => ({
    name: "custom_prompt",
    type: "customtext",
    computeLayoutSize: () => ({ minHeight: 50 }),
});

test("扩展已注册", () => {
    assert.ok(ext);
});

test("loader/Instruct 组默认宽 420", () => {
    registerDef("llama_cpp_vlm_model_loader");
    const node = makeNode("llama_cpp_vlm_model_loader", [makeWidget("model", "a", "combo")]);
    ext.nodeCreated(node);
    assert.deepEqual(node.size, [420, 100]);
});

test("透传/拆分类小节点组默认宽 240", () => {
    registerDef("llama_cpp_unload_model");
    const node = makeNode("llama_cpp_unload_model", []);
    ext.nodeCreated(node);
    assert.deepEqual(node.size, [240, 100]);
});

test("其余节点默认宽 300", () => {
    registerDef("llama_cpp_parameters");
    const node = makeNode("llama_cpp_parameters", [makeWidget("temperature", 1, "number")]);
    ext.nodeCreated(node);
    assert.deepEqual(node.size, [300, 100]);
});

test("内容计算宽超过组默认宽时以计算宽为准", () => {
    registerDef("bboxes_to_segs");
    const node = makeNode("bboxes_to_segs", []);
    node.computeSize = () => [500, 100];
    ext.nodeCreated(node);
    assert.deepEqual(node.size, [500, 100]);
});

test("提示词输入框按默认高度补进节点总高", () => {
    registerDef("llama_cpp_text_instruct");
    const node = makeNode("llama_cpp_text_instruct", [makePromptWidget(), makePromptWidget()]);
    ext.nodeCreated(node);
    assert.deepEqual(node.size, [420, 100 + 2 * (PROMPT_WIDGET_DEFAULT_HEIGHT - 50)]);
});

test("其他 category 的节点不被处理", () => {
    registerDef("other_node", "other-category");
    const node = makeNode("other_node", []);
    ext.nodeCreated(node);
    assert.equal(node.setSizeCalls, 0);
});

test("未注册 comfyClass 的节点不被处理", () => {
    const node = makeNode("unknown_node", []);
    ext.nodeCreated(node);
    assert.equal(node.setSizeCalls, 0);
});
