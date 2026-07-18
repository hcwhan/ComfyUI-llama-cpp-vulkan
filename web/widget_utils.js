// 前端 widget 联动的共用工具 (被 vlm_loader.js / image_instruct.js 引用).

// 隐藏三轨同写, 对齐前端 1.45.20 的两条渲染链路与旧前端:
// canvas 的布局/绘制/命中认实例 hidden 标志; Vue 渲染层 (Nodes 2.0) 只认
// options.hidden, 不读实例标志; type="hidden" + 零高度 computeSize 在
// 当前版本无独立判定点, 作为兼容旧前端的历史轨道保留.
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
    widget.options.hidden = !show;
    widget.type = show ? widget.__origType : "hidden";
    widget.computeSize = show ? widget.__origComputeSize : () => [0, -4];
    return true;
};

// 提示词输入框 (customtext 多行文本 widget) 的默认显示高度: 新建节点时把
// 与各输入框最小高 (前端 fallback 50) 的差额补进节点总高, 前端的弹性空间
// 分配会把富余高度分给弹性 widget, 使输入框呈现为该默认高. 只影响默认值:
// computeSize 不动, 手动拖拽仍可缩到自然下限, 载入工作流恢复保存尺寸
export const PROMPT_WIDGET_DEFAULT_HEIGHT = 120;

export const promptHeightBump = (node) => {
    let extra = 0;
    for (const widget of node.widgets ?? []) {
        if (widget.type !== "customtext") {
            continue;
        }
        const minHeight = widget.computeLayoutSize?.(node)?.minHeight ?? 0;
        extra += Math.max(0, PROMPT_WIDGET_DEFAULT_HEIGHT - minHeight);
    }
    return extra;
};

// 显隐变化后的节点尺寸重排模式, 按调用路径选择:
// - SNAP: 高度贴合到默认高度 (最小计算高 + 提示词输入框补高), 宽度只增
//   不减 (新建节点初次同步; 宽度不缩使其与 node_size.js 的默认宽度设置
//   在任意扩展执行顺序下收敛到同一结果)
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
        const computed = node.computeSize();
        node.setSize([
            Math.max(node.size[0], computed[0]),
            computed[1] + promptHeightBump(node),
        ]);
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
