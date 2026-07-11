# ComfyUI-llama-cpp-vulkan

基于 llama.cpp 的 ComfyUI LLM/VLM 自定义节点插件，使用 Vulkan 实现跨平台 GPU 加速推理。

## 项目概况

- **版本**: 1.2.8
- **核心依赖**: llama-cpp-python (自编译 Vulkan ABI3 wheel, v0.3.41)
- **GPU 后端**: Vulkan (非 CUDA/ROCm，独立于 PyTorch 的 GPU 推理路径)
- **支持平台**: Windows / Linux（预编译 Vulkan wheel 仅覆盖这两个平台）
- **ComfyUI 节点类别**: `llama-cpp-vulkan`

## 目录结构

```
ComfyUI-llama-cpp-vulkan/
  __init__.py                 # 入口：导出 NODE_CLASS_MAPPINGS
  pyproject.toml              # 项目元数据、依赖声明
  requirements.txt            # pip 依赖（含平台条件 llama-cpp-python wheel URL）
  nodes/
    __init__.py               # 节点注册表（12 个节点的映射）
    llm.py                    # 核心：模型加载、推理、GPU 设备检测、会话管理 (~820 行)
    shared.py                 # 公共工具：模型路径、图片编码、预设 prompt、BBox 绘制
    bbox.py                   # BBox 相关节点：JSON 解析、SEGS/MASK 转换
    utils_nodes.py            # 工具节点：JSON 解析、代码块提取、Prompt 增强预设
  support/
    gguf_layers.py            # GGUF 文件解析：读取模型层数 (block_count)
    cqdm.py                   # 进度条封装：同时驱动 ComfyUI ProgressBar 和 tqdm
    prompt_enhancer_preset.py # 18 个 Prompt 增强系统提示词模板
  scripts/
    check_devices.py          # 独立诊断脚本：列出 GGML 后端检测到的所有设备
```

## 节点清单

| 节点 ID | 显示名 | 用途 |
|---------|--------|------|
| `llama_cpp_model_loader` | llama.cpp Model Loader | 加载 GGUF 模型，配置 GPU 设备/上下文长度/显存限制 |
| `llama_cpp_instruct_adv` | llama.cpp Instruct | 文本/图片/视频推理，支持预设 prompt、会话状态、thinking 剥离、生成中途取消 |
| `llama_cpp_parameters` | llama.cpp Parameters | 采样参数配置（temperature/top_k/top_p 等） |
| `llama_cpp_unload_model` | llama.cpp Unload Model | 手动卸载模型释放资源 |
| `llama_cpp_clean_states` | llama.cpp Clean States | 清除保存的会话状态 |
| `parse_json_node` | Parse JSON | 解析 JSON 字符串，按 key 提取值 |
| `json_to_bbox` | JSON to BBoxes | 将 LLM 输出的 JSON 转为 BBox 坐标 |
| `bbox_to_segs` | BBoxes to SEGS | BBox 转 SEGS 格式（兼容 Impact Pack） |
| `bbox_to_mask` | BBoxes to MASK | BBox 转遮罩图 |
| `bboxes_to_bbox` | BBoxes to BBox | 从多组 BBox 中选取特定索引 |
| `remove_code_block` | Unpack Code Block | 去除 LLM 输出中的代码块标记 |
| `PromptEnhancerPreset` | Prompt Enhancer Preset | 18 种 Prompt 增强系统提示词预设 |

## 架构要点

### GPU 设备管理

`llm.py` 通过 ggml C API (ctypes) 直接枚举 Vulkan GPU 设备，区分独显 (GPU) 和核显 (IGPU)。这是独立于 PyTorch/CUDA 的 Vulkan 推理路径。

关键函数链：`_detect_gpu_devices()` -> `_selectable_devices()` -> `_build_gpu_device_choices()` / `_resolve_device_selection()`

选择语义（与 llama.cpp 构建 `model->devices` 的规则对齐）：
- llama.cpp 只把独显按枚举顺序收入设备列表，仅当无独显时才收第一个核显，其余设备无法通过 `main_gpu` 到达，因此下拉框只展示可达设备
- `Auto (独显优先)`：走 llama.cpp 默认行为（`LLAMA_SPLIT_MODE_LAYER`，独显优先，多独显按层切分）
- 显式选择某设备：传 `LLAMA_SPLIT_MODE_NONE` + 该设备在可选列表中的索引，整个模型加载到单卡；`main_gpu` 在 LAYER 模式下会被 llama.cpp 忽略，这是显式选择必须切 NONE 的原因

### 模型生命周期

`LLAMA_CPP_STORAGE` 类管理全局单例模型状态：
- `load_model()`: 加载 GGUF 模型 + 可选的 mmproj（视觉编码器）；纯文本模型（无 mmproj）选择 chat_handler 会在此阶段直接报错
- `clean()`: 释放模型和 chat_handler 资源（不清会话历史）；`clean(all=True)` 额外清除全部会话
- 通过 monkey-patch `mm.unload_all_models` 实现 ComfyUI 模型卸载（前端 Free 按钮 / OOM 处理）时自动清理，只卸模型、保留会话历史

### Chat Handler 动态注册

`chat_handlers` 列表通过 try/except 逐个导入 `llama_cpp.llama_chat_format` 中的 Handler 类，实现运行时能力检测。支持 29+ 种 VLM 模型格式（Qwen/Gemma/GLM/MiniCPM/LLaVA 等）。

### 会话状态管理

`llama_cpp_instruct_adv` 节点通过 `save_states` 参数控制多轮对话。会话历史按 `state_uid` 存储在 `LLAMA_CPP_STORAGE.messages` 中，图片 base64 数据在保存时替换为 1x1 占位图以节省内存。

实现要点：
- system prompt 变化只清当前 `state_uid` 的会话（`clean_state(uid)`），不影响其他会话
- 读取历史时做浅拷贝，推理中断/异常不会把残缺消息写入存储
- 历史为空时自动重建 system 消息（覆盖 `save_states` 从 True 切到 False 的场景）

### 推理输出与中断

- `strip_thinking` 开关（默认开）：用 `_strip_thinking()` 剥离 `<think>...</think>` 推理块；兼容 generation prompt 已注入 `<think>` 导致输出只含闭合标签的情况，未闭合（生成截断）时保持原样
- `_InterruptWatcher`：推理期间守护线程每 200ms 轮询 `mm.processing_interrupted()`，命中后调用 `Llama.abort()` 使生成立即停止；llama-cpp-python 在每次请求开始会 clear abort 事件，因此监视线程命中后持续重复 set 以抗竞态

## 数据流

```
llama_cpp_model_loader  -->  LLAMACPPMODEL (config dict)
                                  |
llama_cpp_parameters  -->  LLAMACPPARAMS (kwargs dict)
                                  |
                                  v
                     llama_cpp_instruct_adv
                       |         |         |
                    STRING    STRING[]    INT
                   (output)  (output_list) (state_uid)
                       |
                       v
              parse_json_node / json_to_bbox / remove_code_block
                       |
                       v
              bbox_to_segs / bbox_to_mask  (下游图像处理)
```

## 修改代码须知

### INPUT_TYPES 字段顺序

ComfyUI 的 widget 值按 `INPUT_TYPES` 中字段的声明顺序序列化。**在字段列表开头或中间插入新字段会导致所有旧工作流的 widget 值错位**。新增字段应追加到末尾，或提供迁移逻辑。

当前 `llama_cpp_model_loader` 的字段顺序：
1. `gpu_device` (后加入的字段，导致旧工作流错位)
2. `model`
3. `mmproj`
4. `chat_handler`
5. `n_ctx`
6. `vram_limit`
7. `image_min_tokens`
8. `image_max_tokens`

`llama_cpp_instruct_adv` 的 `strip_thinking` 是后加字段，已按规范追加到 required 列表末尾（`save_states` 之后），且 `process()` 签名带默认值，旧工作流不受影响。

### 新增 Chat Handler

1. 在 `llm.py` 顶部添加 try/except 导入块
2. 在 `chat_handlers` 列表中追加显示名
3. 在 `LLAMA_CPP_STORAGE.load_model()` 的 `get_chat_handler()` match 语句中添加分支
4. 如有特殊参数（如 `enable_thinking`/`force_reasoning`），在 kwargs 构建处添加条件

### Prompt 增强预设

在 `support/prompt_enhancer_preset.py` 中添加新常量，然后在 `nodes/utils_nodes.py` 的 `PromptEnhancerPreset` 类中注册。

## 依赖

| 包 | 用途 |
|----|------|
| llama-cpp-python | llama.cpp Python 绑定（自编译 Vulkan wheel） |
| diskcache | 缓存（当前代码未直接使用，可能预留） |
| scipy | `gaussian_filter` 用于 BBox 遮罩羽化 |
| numpy | 图像数组操作 |
| pillow | 图像编解码、BBox 绘制 |
| gguf | GGUF 文件元数据读取（备选方案） |
| tqdm | 终端进度条 |

## 已知问题

1. **旧工作流兼容性**: `gpu_device` 字段插入到 `INPUT_TYPES` 开头，导致旧版工作流加载时 widget 值错位（`chat_handler` 收到 `n_ctx` 的值 8192，`n_ctx` 收到 `vram_limit` 的值 -1）。已评估过迁移方案（挪到末尾），决定维持现状避免二次破坏
2. **IS_CHANGED 被注释**: `llama_cpp_model_loader` 的 `IS_CHANGED` 方法被注释掉，且缺少 `gpu_device` 参数，若取消注释需同步更新签名；instruct 节点已有 auto-reload 兜底，功能上不依赖它
3. **单模型实例**: `LLAMA_CPP_STORAGE` 是全局单例，不支持同时加载多个模型
4. **核显不可选（有独显时）**: llama.cpp 的设备收集规则决定了有独显时核显无法通过 `main_gpu` 选中；如需强制核显推理，只能在进程启动前设置 `GGML_VK_VISIBLE_DEVICES` 环境变量（llm.py import 时即初始化 Vulkan，之后设置无效）
5. **import 时初始化 Vulkan**: 设备枚举在插件加载时同步执行（约几百 ms），属有意设计（UI 下拉框需要启动期确定设备列表）
