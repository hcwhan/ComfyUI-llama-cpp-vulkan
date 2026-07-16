// 测试专用: ComfyUI 前端 "scripts/api.js" 的最小替身. fetchApi 只收集调用
// 供断言, 可注入异常模拟网络失败 (路由失败即非 2xx 响应, fetch 不抛异常,
// 被测代码不检查响应, 无需模拟).

export const fetchApiCalls = [];

let fetchApiError = null;

export const setFetchApiError = (error) => {
    fetchApiError = error;
};

export const api = {
    fetchApi: async (path, options) => {
        fetchApiCalls.push({ path, options });
        if (fetchApiError) {
            throw fetchApiError;
        }
        return { status: 200 };
    },
};
