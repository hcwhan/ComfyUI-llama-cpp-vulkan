"""跨语言静态文案: 用户可见但不随 UI 语言切换的常量.

收录两类字符串, 与 language_*.py 分离的原因各自不同:
- 下拉框选项值 (会序列化进工作流的 widget 值): 随语言切换会使已保存
  工作流的对应 widget 值失效 (ComfyUI 对 combo 输入有 value_not_in_list
  前置校验), 故恒为固定值
- 协议串 (被代码逻辑匹配/比较的文本): 生成端与识别端必须同源, 不参与翻译

chat handler 下拉名单不在此处: 显示名即 core/handlers.py 注册表 _HANDLER_SPECS
的 key (功能性标识, 绑定 thinking 三态与构造器, 单一真源).
"""

# ComfyUI 节点分类名 (右键菜单 Add Node 的分类树与搜索面板标签), 全部节点共用
CATEGORY = "llama-cpp-vulkan"

# 控制台日志的统一前缀 (服务器日志过滤标签), 由各日志调用处拼接在消息体前
LOG_PREFIX = "[llama-cpp-vulkan] "

# 下拉框 "未选择" 占位项, 恒在首位作为默认值, 强制用户显式选择
# (model / mmproj / chat_handler / System Prompt Preset 的 preset 四处共用)
NONE_OPTION = "None"

# GPU 设备下拉框 (core/devices.py): Auto 档位标签
AUTO_LABEL = "Auto (GPU First)"
# GPU 设备选项的展示格式, {name}/{desc} 由设备枚举填充, {type} 为 GPU/IGPU;
# resolve_device_selection 按同一模板反查选中设备, 生成端与匹配端同源
DEVICE_LABEL_TEMPLATE = "{name} - {desc} [{type}]"

# image Instruct 的 mode 下拉框 (nodes/instruct/media/image/node_instruct.py)
IMAGE_MODE_EACH = "Per-Image"
IMAGE_MODE_BATCH = "Batch"

# JSON to BBoxes 的 mode 下拉框 (nodes/bbox/node_json_to_bboxes.py);
# 两个 Qwen 值同时是 bbox_utils.json_to_pixel_bboxes 坐标换算的分支匹配值
BBOX_MODE_SIMPLE = "Simple"
BBOX_MODE_QWEN3 = "Qwen3-VL"
BBOX_MODE_QWEN25_VL = "Qwen2.5-VL"

# image Instruct 逐张模式多图输出的前缀行模板, {n} 为图片序号 (从 1 起);
# 生成端 (image Instruct) 与识别端 (shared/text_utils.py 的拆分正则,
# 由本模板派生) 必须同源, 修改模板即同时更新两端
IMAGE_RESULT_SEPARATOR_TEMPLATE = "======== Image {n} ========"
