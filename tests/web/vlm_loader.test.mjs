// web/vlm_loader.js 的单元测试: thinking 开关三态置灰与用户值缓存/恢复,
// image_min/max_tokens 按 handler 名单显隐与区间互钳, onConfigure 再同步
// 且不重排尺寸, 交互切档重排只增不减.
// 能力名单经 nodeData 的 chat_handler widget options 注入 (模拟 /object_info 透传).

import assert from "node:assert/strict";
import { test } from "node:test";

import { extensions, findWidget, loadWebModule, makeNode, makeWidget } from "./harness.mjs";

await loadWebModule("vlm_loader.js");

const NODE_NAME = "llama_cpp_vlm_model_loader";
const ext = extensions.find((e) => e.name === "llama-cpp-vulkan.vlm-loader");

// 三态各一个代表: VisionToggle/VisionForced 支持视觉, AudioNone 为音频专用
const FIXTURE_OPTIONS = {
    thinking_modes: { VisionToggle: "toggle", VisionForced: "forced", AudioNone: "none" },
    image_token_handlers: ["VisionToggle", "VisionForced"],
};

// 模块级能力名单在每个测试开头显式重置, 测试之间互不泄漏
const registerDef = (options = FIXTURE_OPTIONS, name = NODE_NAME) => {
    ext.beforeRegisterNodeDef(null, {
        name,
        input: { required: { chat_handler: [Object.keys(options.thinking_modes ?? {}), options] } },
    });
};

const createNode = (handlerValue, { thinking = false, minTokens = 0, maxTokens = 0 } = {}) => {
    const node = makeNode(NODE_NAME, [
        makeWidget("chat_handler", handlerValue, "combo"),
        makeWidget("thinking", thinking, "toggle"),
        makeWidget("image_min_tokens", minTokens),
        makeWidget("image_max_tokens", maxTokens),
    ]);
    ext.nodeCreated(node);
    return node;
};

const switchHandler = (node, label) => {
    const widget = findWidget(node, "chat_handler");
    widget.value = label;
    widget.callback();
};

test("扩展已注册", () => {
    assert.ok(ext);
});

test("其他 comfyClass 的节点不被处理", () => {
    registerDef();
    const node = makeNode("other_node", [makeWidget("chat_handler", "VisionToggle", "combo")]);
    ext.nodeCreated(node);
    assert.equal(findWidget(node, "chat_handler").callback, undefined);
});

test("缺 chat_handler widget 时直接返回不报错", () => {
    registerDef();
    const node = makeNode(NODE_NAME, [makeWidget("thinking", false, "toggle")]);
    ext.nodeCreated(node);
    assert.equal(findWidget(node, "thinking").disabled, undefined);
});

test("beforeRegisterNodeDef 忽略其他节点的 nodeData", () => {
    registerDef();
    registerDef({ thinking_modes: { VisionToggle: "forced" } }, "other_node");
    const node = createNode("VisionToggle");
    assert.equal(findWidget(node, "thinking").disabled, false);
});

test("新建节点立即同步: forced 档置灰并强制为 true", () => {
    registerDef();
    const node = createNode("VisionForced");
    assert.equal(findWidget(node, "thinking").value, true);
    assert.equal(findWidget(node, "thinking").disabled, true);
});

test("toggle 档切到 none 档: 覆写为 false 并缓存用户值, 切回时恢复", () => {
    registerDef();
    const node = createNode("VisionToggle", { thinking: true });
    switchHandler(node, "AudioNone");
    assert.equal(findWidget(node, "thinking").value, false);
    assert.equal(findWidget(node, "thinking").disabled, true);
    switchHandler(node, "VisionToggle");
    assert.equal(findWidget(node, "thinking").value, true);
    assert.equal(findWidget(node, "thinking").disabled, false);
});

test("forced 与 none 档之间切换不重复缓存, toggle 档的用户值保留", () => {
    registerDef();
    const node = createNode("VisionToggle", { thinking: true });
    switchHandler(node, "VisionForced");
    switchHandler(node, "AudioNone");
    switchHandler(node, "VisionToggle");
    assert.equal(findWidget(node, "thinking").value, true);
});

test("未知 label 归一为 toggle 档: 不置灰, 恢复缓存的用户值", () => {
    registerDef();
    const node = createNode("VisionToggle", { thinking: true });
    switchHandler(node, "VisionForced");
    switchHandler(node, "UnknownHandler");
    assert.equal(findWidget(node, "thinking").value, true);
    assert.equal(findWidget(node, "thinking").disabled, false);
});

test("thinking_modes 缺失时 thinking 不被联动", () => {
    registerDef({ image_token_handlers: ["VisionToggle"] });
    const node = createNode("VisionForced", { thinking: false });
    assert.equal(findWidget(node, "thinking").value, false);
    assert.equal(findWidget(node, "thinking").disabled, undefined);
});

test("image token 字段按 handler 名单显隐, 值保留", () => {
    registerDef();
    const node = createNode("AudioNone", { minTokens: 10, maxTokens: 1280 });
    assert.equal(findWidget(node, "image_min_tokens").type, "hidden");
    assert.equal(findWidget(node, "image_max_tokens").type, "hidden");
    switchHandler(node, "VisionToggle");
    assert.equal(findWidget(node, "image_min_tokens").type, "number");
    assert.equal(findWidget(node, "image_min_tokens").value, 10);
    assert.equal(findWidget(node, "image_max_tokens").value, 1280);
});

test("setSize 仅在显隐实际变化时调用", () => {
    registerDef();
    const node = createNode("VisionToggle");
    assert.equal(node.setSizeCalls, 0);
    switchHandler(node, "AudioNone");
    assert.equal(node.setSizeCalls, 1);
    switchHandler(node, "VisionForced");
    assert.equal(node.setSizeCalls, 2);
    switchHandler(node, "VisionToggle");
    assert.equal(node.setSizeCalls, 2);
});

test("区间互钳: min 抬高越过 max 时 max 跟随", () => {
    registerDef();
    const node = createNode("VisionToggle", { minTokens: 10, maxTokens: 100 });
    const minWidget = findWidget(node, "image_min_tokens");
    minWidget.value = 200;
    minWidget.callback();
    assert.equal(findWidget(node, "image_max_tokens").value, 200);
});

test("区间互钳: max 压低到 min 之下时 min 跟随", () => {
    registerDef();
    const node = createNode("VisionToggle", { minTokens: 100, maxTokens: 200 });
    const maxWidget = findWidget(node, "image_max_tokens");
    maxWidget.value = 50;
    maxWidget.callback();
    assert.equal(findWidget(node, "image_min_tokens").value, 50);
});

test("max=0 视为未设置: 不参与互钳", () => {
    registerDef();
    const node = createNode("VisionToggle", { minTokens: 100, maxTokens: 0 });
    const minWidget = findWidget(node, "image_min_tokens");
    minWidget.value = 500;
    minWidget.callback();
    assert.equal(findWidget(node, "image_max_tokens").value, 0);
    const maxWidget = findWidget(node, "image_max_tokens");
    maxWidget.value = 0;
    maxWidget.callback();
    assert.equal(minWidget.value, 500);
});

test("onConfigure 恢复序列化值后再同步", () => {
    registerDef();
    const node = createNode("VisionToggle");
    // 模拟载入工作流: configure 把 widget 值恢复为序列化内容后触发 onConfigure
    findWidget(node, "chat_handler").value = "VisionForced";
    findWidget(node, "thinking").value = false;
    node.onConfigure();
    assert.equal(findWidget(node, "thinking").value, true);
    assert.equal(findWidget(node, "thinking").disabled, true);
});

test("onConfigure 显隐变化时不重排尺寸", () => {
    registerDef();
    // 保存为视觉 handler 的工作流: nodeCreated 首次同步以默认档隐藏
    // image token 字段, onConfigure 按恢复值重新显示 (显隐必然变化)
    const node = createNode("AudioNone");
    const callsAfterCreate = node.setSizeCalls;
    node.size = [321, 456];
    findWidget(node, "chat_handler").value = "VisionToggle";
    node.onConfigure();
    assert.equal(findWidget(node, "image_min_tokens").type, "number");
    // 回归: 旧实现在 onConfigure 内 setSize(computeSize()),
    // 把刚恢复的序列化尺寸压回最小计算值
    assert.equal(node.setSizeCalls, callsAfterCreate);
    assert.deepEqual(node.size, [321, 456]);
});

test("交互切档重排只增不减: 用户拉大的尺寸不被缩小", () => {
    registerDef();
    const node = createNode("VisionToggle");
    node.size = [300, 400];
    switchHandler(node, "AudioNone");
    assert.equal(findWidget(node, "image_min_tokens").type, "hidden");
    assert.deepEqual(node.size, [300, 400]);
});

test("原有 widget callback 仍被调用", () => {
    registerDef();
    let called = 0;
    const handlerWidget = makeWidget("chat_handler", "VisionToggle", "combo");
    handlerWidget.callback = () => {
        called += 1;
    };
    const node = makeNode(NODE_NAME, [handlerWidget, makeWidget("thinking", false, "toggle")]);
    ext.nodeCreated(node);
    handlerWidget.callback();
    assert.equal(called, 1);
});
