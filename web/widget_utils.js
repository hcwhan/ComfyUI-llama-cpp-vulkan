// 前端 widget 联动的共用工具 (被 vlm_loader.js / image_instruct.js 引用).

// 隐藏需 hidden 标志与 type="hidden" + 零高度 computeSize 双管齐下:
// 新前端的 Vue 渲染层认 hidden 标志, canvas 布局认 type/computeSize.
// 不动 serialize, 隐藏值仍按声明序序列化. 返回是否发生切换
export const toggleWidget = (widget, show) => {
    if (widget.__origType === undefined) {
        widget.__origType = widget.type;
        widget.__origComputeSize = widget.computeSize;
    }
    const currentlyShown = !widget.hidden && widget.type !== "hidden";
    if (show === currentlyShown) {
        return false;
    }
    widget.hidden = !show;
    widget.type = show ? widget.__origType : "hidden";
    widget.computeSize = show ? widget.__origComputeSize : () => [0, -4];
    return true;
};
