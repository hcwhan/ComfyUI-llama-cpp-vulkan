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

// 显隐变化后的节点尺寸重排模式, 按调用路径选择:
// - SNAP: 贴合到最小计算尺寸 (新建节点初次同步, 收紧默认尺寸)
// - GROW: 只增不减, 仅在装不下新显示的 widget 时放大 (交互切档,
//   保留用户手动拉大的尺寸, 代价是切回少 widget 档位时底部留白不回收)
// - NONE: 不动尺寸 (configure 载入路径: LGraphNode.configure 先恢复序列化
//   尺寸再调 onConfigure, 且保存时的显隐状态与恢复后一致, 此处重排反而会
//   把刚恢复的用户尺寸压回最小计算尺寸; undo/redo 重新 configure 同理)
export const REFLOW_SNAP = "snap";
export const REFLOW_GROW = "grow";
export const REFLOW_NONE = "none";

export const reflowNode = (node, mode) => {
    if (mode === REFLOW_SNAP) {
        node.setSize(node.computeSize());
    } else if (mode === REFLOW_GROW) {
        const computed = node.computeSize();
        node.setSize([
            Math.max(node.size[0], computed[0]),
            Math.max(node.size[1], computed[1]),
        ]);
    } else if (mode === REFLOW_NONE) {
        // 显式空分支: 该模式明确不动尺寸
    }
};
