// 测试引导: 把 web/*.js 对 "../../scripts/app.js" 与 "../../scripts/api.js"
// 的 import 重定向到替身 (module.registerHooks 同步钩子, Node >= 22.15),
// 并提供 widget/node 工厂.
// web 侧 JS 只操作鸭子类型的普通对象, 无 DOM 依赖, 可在 Node 进程直接驱动.

import { registerHooks } from "node:module";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const _here = path.dirname(fileURLToPath(import.meta.url));
const _appStubUrl = pathToFileURL(path.join(_here, "comfy_app_stub.mjs")).href;
const _apiStubUrl = pathToFileURL(path.join(_here, "comfy_api_stub.mjs")).href;
const _webDir = path.resolve(_here, "..", "..", "web");

registerHooks({
    resolve(specifier, context, nextResolve) {
        if (specifier.endsWith("scripts/app.js")) {
            return { url: _appStubUrl, shortCircuit: true };
        }
        if (specifier.endsWith("scripts/api.js")) {
            return { url: _apiStubUrl, shortCircuit: true };
        }
        return nextResolve(specifier, context);
    },
});

// 动态 import 保证钩子先于 web 模块加载生效; 模块级注册 (registerExtension)
// 在首次 import 时执行一次, extensions 数组按 import 顺序收集
export const loadWebModule = async (name) => {
    return import(pathToFileURL(path.join(_webDir, name)).href);
};

export { extensions, settingValues } from "./comfy_app_stub.mjs";
export { fetchApiCalls, setFetchApiError } from "./comfy_api_stub.mjs";

// 与 ComfyUI 前端 widget 对象的最小对齐: name/value/type/computeSize/callback
export const makeWidget = (name, value, type = "number") => ({
    name,
    value,
    type,
    computeSize: () => [100, 20],
    callback: undefined,
});

// setSize 计数用于断言 "仅在显隐实际变化时重排" 的行为;
// size 随 setSize 同步更新 (对齐 LGraphNode), 供重排模式断言尺寸结果
export const makeNode = (comfyClass, widgets) => ({
    comfyClass,
    widgets,
    size: [200, 100],
    setSizeCalls: 0,
    setSize(size) {
        this.setSizeCalls += 1;
        this.size = size;
    },
    computeSize: () => [200, 100],
});

export const findWidget = (node, name) => node.widgets.find((w) => w.name === name);
