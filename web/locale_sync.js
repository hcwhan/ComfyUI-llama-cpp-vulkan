// 页面加载时把前端实际显示语言上报到后端一次, 持久化供下次启动的语言解析
// 兜底 (Comfy.Locale 从未设置时前端按浏览器语言自动检测, 服务端无从得知).
// 取值走 setting.get: 已保存时返回保存值, 未保存时返回前端按 navigator
// 解析的默认值, 恒为实际显示语言. 不做变化监听: 用户改语言时 ComfyUI 自己
// 会持久化 Comfy.Locale, 它在下次启动的解析中优先级最高 (见 i18n/lang.py).
// 上报失败静默忽略 (JS 失效只损失下次启动的兜底, 行为正确性由 Python 侧保证).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "llama-cpp-vulkan.locale-sync",
    async setup() {
        const locale = app.extensionManager.setting.get("Comfy.Locale");
        if (typeof locale !== "string" || !locale) {
            return;
        }
        try {
            await api.fetchApi("/llama_cpp_vulkan/frontend_locale", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ locale }),
            });
        } catch {
            // 网络/路由异常静默忽略
        }
    },
});
