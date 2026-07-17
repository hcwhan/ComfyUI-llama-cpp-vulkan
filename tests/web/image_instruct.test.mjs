// web/image_instruct.js 的单元测试: increment_seed 仅在 Per-Image 档显示,
// max_size 仅在 Batch 档显示, 显隐切换保留值, 自定义 key (each_mode_value /
// batch_mode_value) 任一缺失时不联动, onConfigure 再同步且不重排尺寸,
// 交互切档重排只增不减.

import assert from "node:assert/strict";
import { test } from "node:test";

import { extensions, findWidget, loadWebModule, makeNode, makeWidget } from "./harness.mjs";

await loadWebModule("image_instruct.js");

const NODE_NAME = "llama_cpp_image_instruct";
const ext = extensions.find((e) => e.name === "llama-cpp-vulkan.image-instruct");

// 与 common_static.py 的 IMAGE_MODE_EACH/IMAGE_MODE_BATCH 对应的选项值形态
const MODE_EACH = "Per-Image";
const MODE_BATCH = "Batch";

// 模块级 eachModeValue/batchModeValue 在每个测试开头显式重置, 测试之间互不泄漏
const registerDef = (options = { each_mode_value: MODE_EACH, batch_mode_value: MODE_BATCH }) => {
    ext.beforeRegisterNodeDef(null, {
        name: NODE_NAME,
        input: { required: { mode: [[MODE_EACH, MODE_BATCH], options] } },
    });
};

const createNode = (modeValue) => {
    const node = makeNode(NODE_NAME, [
        makeWidget("mode", modeValue, "combo"),
        makeWidget("increment_seed", false, "toggle"),
        makeWidget("max_size", 256, "number"),
    ]);
    ext.nodeCreated(node);
    return node;
};

const switchMode = (node, value) => {
    const widget = findWidget(node, "mode");
    widget.value = value;
    widget.callback();
};

test("扩展已注册", () => {
    assert.ok(ext);
});

test("Per-Image 档创建: max_size 隐藏, increment_seed 可见", () => {
    registerDef();
    const node = createNode(MODE_EACH);
    assert.equal(findWidget(node, "max_size").type, "hidden");
    assert.equal(findWidget(node, "increment_seed").type, "toggle");
    assert.equal(node.setSizeCalls, 1);
});

test("Batch 档创建: max_size 可见, increment_seed 隐藏", () => {
    registerDef();
    const node = createNode(MODE_BATCH);
    assert.equal(findWidget(node, "max_size").type, "number");
    assert.equal(findWidget(node, "increment_seed").type, "hidden");
    assert.equal(node.setSizeCalls, 1);
});

test("切换 mode: 显隐互补随动且值保留, 无变化时不重排", () => {
    registerDef();
    const node = createNode(MODE_BATCH);
    switchMode(node, MODE_EACH);
    assert.equal(findWidget(node, "max_size").type, "hidden");
    assert.equal(findWidget(node, "increment_seed").type, "toggle");
    assert.equal(node.setSizeCalls, 2);
    switchMode(node, MODE_EACH);
    assert.equal(node.setSizeCalls, 2);
    switchMode(node, MODE_BATCH);
    assert.equal(findWidget(node, "max_size").type, "number");
    assert.equal(findWidget(node, "max_size").value, 256);
    assert.equal(findWidget(node, "increment_seed").type, "hidden");
    assert.equal(findWidget(node, "increment_seed").value, false);
    assert.equal(node.setSizeCalls, 3);
});

test("自定义 key 全缺失时不联动", () => {
    registerDef({});
    const node = createNode(MODE_EACH);
    assert.equal(findWidget(node, "max_size").type, "number");
    assert.equal(findWidget(node, "increment_seed").type, "toggle");
    assert.equal(findWidget(node, "mode").callback, undefined);
});

test("自定义 key 任一缺失时不联动", () => {
    registerDef({ each_mode_value: MODE_EACH });
    const nodeA = createNode(MODE_BATCH);
    assert.equal(findWidget(nodeA, "increment_seed").type, "toggle");
    registerDef({ batch_mode_value: MODE_BATCH });
    const nodeB = createNode(MODE_EACH);
    assert.equal(findWidget(nodeB, "max_size").type, "number");
});

test("其他 comfyClass 的节点不被处理", () => {
    registerDef();
    const node = makeNode("other_node", [
        makeWidget("mode", MODE_EACH, "combo"),
        makeWidget("increment_seed", false, "toggle"),
        makeWidget("max_size", 256, "number"),
    ]);
    ext.nodeCreated(node);
    assert.equal(findWidget(node, "max_size").type, "number");
    assert.equal(findWidget(node, "increment_seed").type, "toggle");
});

test("onConfigure 恢复序列化值后再同步, 不重排尺寸", () => {
    registerDef();
    const node = createNode(MODE_BATCH);
    // 模拟载入工作流: configure 先恢复序列化尺寸与 widget 值, 再触发 onConfigure
    const callsAfterCreate = node.setSizeCalls;
    node.size = [321, 456];
    findWidget(node, "mode").value = MODE_EACH;
    node.onConfigure();
    assert.equal(findWidget(node, "max_size").type, "hidden");
    assert.equal(findWidget(node, "increment_seed").type, "toggle");
    // 回归: 旧实现在 onConfigure 内 setSize(computeSize()),
    // 把刚恢复的序列化尺寸压回最小计算值
    assert.equal(node.setSizeCalls, callsAfterCreate);
    assert.deepEqual(node.size, [321, 456]);
});

test("交互切档重排只增不减: 用户拉大的尺寸不被缩小", () => {
    registerDef();
    const node = createNode(MODE_BATCH);
    node.size = [300, 400];
    switchMode(node, MODE_EACH);
    assert.equal(findWidget(node, "max_size").type, "hidden");
    assert.deepEqual(node.size, [300, 400]);
});

test("原有 widget callback 仍被调用", () => {
    registerDef();
    let called = 0;
    const modeWidget = makeWidget("mode", MODE_BATCH, "combo");
    modeWidget.callback = () => {
        called += 1;
    };
    const node = makeNode(NODE_NAME, [
        modeWidget,
        makeWidget("increment_seed", false, "toggle"),
        makeWidget("max_size", 256, "number"),
    ]);
    ext.nodeCreated(node);
    modeWidget.callback();
    assert.equal(called, 1);
});
