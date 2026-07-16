// web/widget_utils.js 的单元测试: toggleWidget 双轨隐藏语义
// (hidden 标志 + type/computeSize), 原始值缓存与无变化时的短路返回.

import assert from "node:assert/strict";
import { test } from "node:test";

import { makeWidget } from "./harness.mjs";

const { toggleWidget } = await import("../../web/widget_utils.js");

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
