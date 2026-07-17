// web/widget_utils.js 的单元测试: toggleWidget 双轨隐藏语义
// (hidden 标志 + type/computeSize), 原始值缓存与无变化时的短路返回;
// reflowNode 三种重排模式 (SNAP / 只增不减 / 不动) 与提示词输入框补高.

import assert from "node:assert/strict";
import { test } from "node:test";

import { makeNode, makeWidget } from "./harness.mjs";

const { toggleWidget, reflowNode, promptHeightBump, PROMPT_WIDGET_DEFAULT_HEIGHT, REFLOW_SNAP, REFLOW_GROW, REFLOW_NONE } = await import("../../web/widget_utils.js");

// 与前端 customtext DOM widget 的最小对齐: type + computeLayoutSize
const makePromptWidget = (minHeight = 50) => ({
    name: "custom_prompt",
    type: "customtext",
    computeLayoutSize: () => ({ minHeight }),
});

test("隐藏: hidden 标志与 type/computeSize 双轨同时生效", () => {
    const widget = makeWidget("max_size", 256, "number");
    const changed = toggleWidget(widget, false);
    assert.equal(changed, true);
    assert.equal(widget.hidden, true);
    assert.equal(widget.type, "hidden");
    assert.deepEqual(widget.computeSize(), [0, -4]);
});

test("重新显示: 恢复原始 type 与 computeSize", () => {
    const widget = makeWidget("max_size", 256, "number");
    const origComputeSize = widget.computeSize;
    toggleWidget(widget, false);
    const changed = toggleWidget(widget, true);
    assert.equal(changed, true);
    assert.equal(widget.hidden, false);
    assert.equal(widget.type, "number");
    assert.equal(widget.computeSize, origComputeSize);
});

test("无变化时短路返回 false 且不改显隐状态", () => {
    const widget = makeWidget("max_size", 256, "number");
    assert.equal(toggleWidget(widget, true), false);
    assert.equal(widget.type, "number");
    toggleWidget(widget, false);
    assert.equal(toggleWidget(widget, false), false);
    assert.equal(widget.type, "hidden");
});

test("原始 type 只缓存一次, 多轮切换不被 hidden 覆盖", () => {
    const widget = makeWidget("image_min_tokens", 10, "number");
    toggleWidget(widget, false);
    toggleWidget(widget, true);
    toggleWidget(widget, false);
    toggleWidget(widget, true);
    assert.equal(widget.type, "number");
    assert.equal(widget.__origType, "number");
});

test("隐藏不触碰 value (序列化值保持)", () => {
    const widget = makeWidget("image_max_tokens", 1280, "number");
    toggleWidget(widget, false);
    assert.equal(widget.value, 1280);
});

test("reflowNode SNAP: 高度贴合计算值, 宽度只增不减", () => {
    const node = makeNode("any", []);
    node.size = [300, 400];
    reflowNode(node, REFLOW_SNAP);
    assert.equal(node.setSizeCalls, 1);
    assert.deepEqual(node.size, [300, 100]);
    node.size = [150, 400];
    reflowNode(node, REFLOW_SNAP);
    assert.deepEqual(node.size, [200, 100]);
});

test("reflowNode SNAP: 提示词输入框按默认高度补进节点总高", () => {
    const node = makeNode("any", [makePromptWidget(), makePromptWidget(), makeWidget("mode", "x", "combo")]);
    reflowNode(node, REFLOW_SNAP);
    assert.deepEqual(node.size, [200, 100 + 2 * (PROMPT_WIDGET_DEFAULT_HEIGHT - 50)]);
});

test("promptHeightBump: 非 customtext 不参与, 最小高不低于默认高时不补", () => {
    const node = makeNode("any", [makeWidget("seed", 0, "number"), makePromptWidget(PROMPT_WIDGET_DEFAULT_HEIGHT + 20)]);
    assert.equal(promptHeightBump(node), 0);
});

test("reflowNode GROW: 已放大的尺寸不被缩小", () => {
    const node = makeNode("any", []);
    node.size = [300, 400];
    reflowNode(node, REFLOW_GROW);
    assert.deepEqual(node.size, [300, 400]);
});

test("reflowNode GROW: 装不下时按维度放大到计算值", () => {
    const node = makeNode("any", []);
    node.size = [300, 80];
    node.computeSize = () => [350, 120];
    reflowNode(node, REFLOW_GROW);
    assert.deepEqual(node.size, [350, 120]);
    node.size = [500, 100];
    reflowNode(node, REFLOW_GROW);
    assert.deepEqual(node.size, [500, 120]);
});

test("reflowNode NONE 与未知模式: 不调用 setSize", () => {
    const node = makeNode("any", []);
    node.size = [300, 400];
    reflowNode(node, REFLOW_NONE);
    reflowNode(node, "unknown");
    assert.equal(node.setSizeCalls, 0);
    assert.deepEqual(node.size, [300, 400]);
});
