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
    llm.py                    # 核心：模型加载、推理、GPU 设备检测、会话管理 (~716 行)
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
| `llama_cpp_instruct_adv` | llama.cpp Instruct | 文本/图片/视频推理，支持多种预设 prompt 和会话状态 |
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

`llm.py` 通过 ggml C API (ctypes) 直接枚举 Vulkan GPU 设备，区分独显 (GPU) 和核显 (IGPU)。`Auto` 模式优先选择独显。这是独立于 PyTorch/CUDA 的 Vulkan 推理路径。

关键函数链：`_detect_gpu_devices()` -> `_build_gpu_device_choices()` -> `_resolve_main_gpu()`

### 模型生命周期

`LLAMA_CPP_STORAGE` 类管理全局单例模型状态：
- `load_model()`: 加载 GGUF 模型 + 可选的 mmproj（视觉编码器）
- `clean()`: 释放模型和 chat_handler 资源
- 通过 monkey-patch `mm.unload_all_models` 实现 ComfyUI 模型卸载时自动清理

### Chat Handler 动态注册

`chat_handlers` 列表通过 try/except 逐个导入 `llama_cpp.llama_chat_format` 中的 Handler 类，实现运行时能力检测。支持 29+ 种 VLM 模型格式（Qwen/Gemma/GLM/MiniCPM/LLaVA 等）。

### 会话状态管理

`llama_cpp_instruct_adv` 节点通过 `save_states` 参数控制多轮对话。会话历史按 `state_uid` 存储在 `LLAMA_CPP_STORAGE.messages` 中，图片 base64 数据在保存时替换为 1x1 占位图以节省内存。

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

1. **旧工作流兼容性**: `gpu_device` 字段插入到 `INPUT_TYPES` 开头，导致旧版工作流加载时 widget 值错位（`chat_handler` 收到 `n_ctx` 的值 8192，`n_ctx` 收到 `vram_limit` 的值 -1）
2. **IS_CHANGED 被注释**: `llama_cpp_model_loader` 的 `IS_CHANGED` 方法被注释掉，且缺少 `gpu_device` 参数，若取消注释需同步更新签名
3. **单模型实例**: `LLAMA_CPP_STORAGE` 是全局单例，不支持同时加载多个模型
