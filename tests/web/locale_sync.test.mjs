// web/locale_sync.js 的单元测试: setup 时上报一次 Comfy.Locale 当前生效值,
// 值缺失/非字符串/为空时不上报, 上报失败静默不抛出.

import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import { extensions, fetchApiCalls, loadWebModule, setFetchApiError, settingValues } from "./harness.mjs";

await loadWebModule("locale_sync.js");

const ext = extensions.find((e) => e.name === "llama-cpp-vulkan.locale-sync");

beforeEach(() => {
    fetchApiCalls.length = 0;
    setFetchApiError(null);
    delete settingValues["Comfy.Locale"];
});

test("setup 上报当前语言到 frontend_locale 路由", async () => {
    settingValues["Comfy.Locale"] = "zh";
    await ext.setup();
    assert.equal(fetchApiCalls.length, 1);
    const { path, options } = fetchApiCalls[0];
    assert.equal(path, "/llama_cpp_vulkan/frontend_locale");
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body), { locale: "zh" });
});

test("语言值缺失时不上报", async () => {
    await ext.setup();
    assert.equal(fetchApiCalls.length, 0);
});

test("语言值非字符串或为空时不上报", async () => {
    settingValues["Comfy.Locale"] = 42;
    await ext.setup();
    settingValues["Comfy.Locale"] = "";
    await ext.setup();
    assert.equal(fetchApiCalls.length, 0);
});

test("上报失败静默不抛出", async () => {
    settingValues["Comfy.Locale"] = "en";
    setFetchApiError(new Error("network down"));
    await ext.setup();
    assert.equal(fetchApiCalls.length, 1);
});
