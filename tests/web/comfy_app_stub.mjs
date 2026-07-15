// 测试专用: ComfyUI 前端 "scripts/app.js" 的最小替身 (角色等同 Python 侧
// comfy_stubs.py). registerExtension 只收集扩展对象, 供测试直接驱动
// beforeRegisterNodeDef / nodeCreated 等生命周期钩子.

export const extensions = [];

export const app = {
    registerExtension: (ext) => {
        extensions.push(ext);
    },
};
