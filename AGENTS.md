# ComfyUI-llama-cpp-vulkan

基于 llama.cpp 的 ComfyUI LLM/VLM 自定义节点插件，使用 Vulkan 实现跨平台 GPU 加速推理。

## 项目概况

- **版本**: 1.5.0
- **核心依赖**: llama-cpp-python (自编译 Vulkan ABI3 wheel, v0.3.41)
- **GPU 后端**: Vulkan (非 CUDA/ROCm，独立于 PyTorch 的 GPU 推理路径)
- **支持平台**: Windows / Linux（预编译 Vulkan wheel 仅覆盖这两个平台；Linux 为 manylinux_2_31，要求 glibc >= 2.31）
- **ComfyUI 节点类别**: `llama-cpp-vulkan`

## 目录结构

```
ComfyUI-llama-cpp-vulkan/
  __init__.py                 # 入口：导出 NODE_CLASS_MAPPINGS
  pyproject.toml              # 项目元数据、依赖声明
  requirements.txt            # pip 依赖（含平台条件 llama-cpp-python wheel URL）
  .github/workflows/
    build-vulkan-wheels-abi3.yml  # CI：构建/发布双平台 Vulkan ABI3 wheel
  nodes/
    __init__.py               # 节点注册表（11 个节点的映射）
    llm.py                    # 核心：模型加载、推理 (~520 行)
    devices.py                # Vulkan GPU 设备检测与选择（ggml C API / ctypes）
    handlers.py               # Chat handler 注册表（30 种 VLM 格式）
    shared.py                 # 公共工具：模型路径、图片/音频编码、预设 prompt、BBox 坐标换算与绘制
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
| `llama_cpp_instruct_adv` | llama.cpp Instruct | 文本/图片/音频/视频推理，支持预设 prompt、thinking 剥离、生成中途取消 |
| `llama_cpp_parameters` | llama.cpp Parameters | 采样参数配置（temperature/top_k/top_p 等） |
| `llama_cpp_unload_model` | llama.cpp Unload Model | 手动卸载模型释放资源 |
| `parse_json_node` | Parse JSON | 解析 JSON 字符串，按 key 提取值 |
| `json_to_bbox` | JSON to BBoxes | 将 LLM 输出的 JSON 转为 BBox 坐标 |
| `bbox_to_segs` | BBoxes to SEGS | BBox 转 SEGS 格式（兼容 Impact Pack） |
| `bbox_to_mask` | BBoxes to MASK | BBox 转遮罩图 |
| `bboxes_to_bbox` | BBoxes to BBox | 从多组 BBox 中选取特定索引 |
| `remove_code_block` | Unpack Code Block | 去除 LLM 输出中的代码块标记 |
| `PromptEnhancerPreset` | Prompt Enhancer Preset | 18 种 Prompt 增强系统提示词预设 |

## 架构要点

### GPU 设备管理

`nodes/devices.py` 通过 ggml C API (ctypes) 直接枚举 Vulkan GPU 设备，区分独显 (GPU) 和核显 (IGPU)。这是独立于 PyTorch/CUDA 的 Vulkan 推理路径。

关键函数链：`_detect_gpu_devices()` -> `_selectable_devices()` -> `gpu_device_choices` / `resolve_device_selection()`

选择语义（与 llama.cpp 构建 `model->devices` 的规则对齐）：
- llama.cpp 只把独显按枚举顺序收入设备列表，仅当无独显时才收第一个核显，其余设备无法通过 `main_gpu` 到达，因此下拉框只展示可达设备
- `Auto (独显优先)`：走 llama.cpp 默认行为（`LLAMA_SPLIT_MODE_LAYER`，独显优先，多独显按层切分）
- 显式选择某设备：传 `LLAMA_SPLIT_MODE_NONE` + 该设备在可选列表中的索引，整个模型加载到单卡；`main_gpu` 在 LAYER 模式下会被 llama.cpp 忽略，这是显式选择必须切 NONE 的原因

### 模型生命周期

`LLAMA_CPP_STORAGE` 类管理全局单例模型状态：
- 懒加载：`llama_cpp_model_loader` 只调用 `_resolve_config()` 做快速失败校验（模型/mmproj 路径存在、mmproj 与 chat_handler 配对合法）并返回 config，实际加载由 `llama_cpp_instruct_adv` 按需触发；多组 loader+instruct 交错时避免全局单例被 loader 反复挤占
- `load_model()`: 先 `_resolve_config()` 校验再卸载旧模型（无效配置不影响已加载的模型），随后加载 GGUF 模型 + 可选的 mmproj（视觉编码器）
- `clean()`: 释放模型和 chat_handler 资源
- 通过 monkey-patch `mm.unload_all_models` 实现 ComfyUI 模型卸载（前端 Free 按钮 / OOM 处理）时自动清理
- `vram_limit` 折算 `n_gpu_layers` 集中在 `_estimate_n_gpu_layers()`：按 GGUF 层数（`support/gguf_layers.py` 手写解析 `block_count`，命中即返回避免解析 tokenizer 元数据）均摊文件体积，乘经验系数 `_VRAM_OVERHEAD_FACTOR`(1.55) 估算每层显存，mmproj 体积先从预算中扣除

### Chat Handler 注册表

`nodes/handlers.py` 中的 `_HANDLER_SPECS` 表集中定义全部 handler：显示名 -> (类名, thinking 开关参数名)。启动时 `_resolve_handlers()` 用 `getattr` 对照 handler 模块解析类名，缺失的类打 warning 并从下拉框剔除（不静默）。支持 30 种 VLM 模型格式（Qwen/Gemma/GLM/MiniCPM/LLaVA 等）。

Handler 模块优先取 `llama_cpp.llama_multimodal`（JamePeng 分支，requirements.txt 固定的 wheel），官方构建无此模块时回退 `llama_cpp.llama_chat_format`。mmproj 路径统一用 `clip_model_path` 键传入：官方构建只认这个名字，JamePeng 构建把它作为 `mmproj_path` 的兼容别名接受。

### 多模态输入

- 图片：`images` 输入按 `inference_mode`（one by one / images / video）构造 `image_url` 内容项；images/video 模式多帧时缩放到 `max_size`，单帧保持原分辨率
- 音频：`audio` 可选输入（ComfyUI `AUDIO` dict）由 `shared.audio2base64()` 均值混为单声道 16-bit WAV，以 `input_audio` 内容项注入（重采样由 llama.cpp 的 mtmd 解码端完成），服务 Qwen3-ASR 等音频 handler；仅 MTMD 构建可用——官方构建的旧式 handler 会静默忽略音频项，`_append_audio()` 在节点层直接报错拦截

### 推理输出与中断

- 无会话状态：每次执行都是全新的一次性请求（system prompt + 本次提问），不保留任何跨执行的对话历史
- `strip_thinking` 开关（默认开）：用 `_strip_thinking()` 剥离 `<think>...</think>` 推理块；兼容 generation prompt 已注入 `<think>` 导致输出只含闭合标签的情况，未闭合（生成截断）时保持原样
- `_InterruptWatcher`：推理期间守护线程每 200ms 轮询 `mm.processing_interrupted()`，命中后调用 `Llama.abort()` 使生成立即停止；llama-cpp-python 在每次请求开始会 clear abort 事件，因此监视线程命中后持续重复 set 以抗竞态
- 每次请求结束后按 `_is_hybrid_arch()`（`_model.is_hybrid()`/`is_recurrent()` C API）判断是否整体重置 KV cache：hybrid/recurrent 架构（Qwen3.5、LFM2 系等）的线性注意力状态无法跨请求前缀复用；纯 SWA 模型（Gemma3）不受影响

## 数据流

```
llama_cpp_model_loader  -->  LLAMACPPMODEL (config dict)
                                  |
llama_cpp_parameters  -->  LLAMACPPARAMS (kwargs dict)
                                  |
                                  v
                     llama_cpp_instruct_adv
                       |         |
                    STRING    STRING[]
                   (output)  (output_list)
                       |
                       v
              parse_json_node / json_to_bbox / remove_code_block
                       |
                       v
              bbox_to_segs / bbox_to_mask  (下游图像处理)
```

## 修改代码须知

### INPUT_TYPES 字段顺序

ComfyUI 的 widget 值按 `INPUT_TYPES` 中字段的声明顺序序列化。

当前 `llama_cpp_model_loader` 的字段顺序：
1. `gpu_device`
2. `model`
3. `mmproj`
4. `chat_handler`
5. `n_ctx`
6. `vram_limit`
7. `image_min_tokens`
8. `image_max_tokens`

### 新增 Chat Handler

在 `nodes/handlers.py` 的 `_HANDLER_SPECS` 表中加一行：`"显示名": ("类名", thinking开关参数名或None)`。显示名含 "-Thinking" 后缀时加载器自动把开关参数设为 True。注意 thinking 参数名必须被该类 `__init__` 接受（基类会对未知 kwargs 抛 TypeError），如 GLM41VChatHandler 不接受 `enable_thinking`。

### Prompt 增强预设

在 `support/prompt_enhancer_preset.py` 中添加新常量，并在文件末尾的 `PRESETS` dict 中加一行（dict 顺序即 UI 下拉框顺序）。

### Wheel 构建与发布 (CI)

`.github/workflows/build-vulkan-wheels-abi3.yml` 手动触发（workflow_dispatch，输入 JamePeng/llama-cpp-python 的 ref），并行构建 Windows + Linux 两个 ABI3 wheel 并自动发布 GitHub Release。要点：

- `+vulkan` 本地版本在构建前注入 `llama_cpp/__init__.py` 的 `__version__`，wheel 文件名与 METADATA 天然一致（不做构建后重命名）
- 两平台均开启 `GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS`（CPU 后端按指令集运行时分发）
- Windows：Vulkan SDK 走 action 缓存（stripdown 后缓存），编译走 sccache；冷缓存约 25 分钟，热缓存约 5 分钟
- Linux：Vulkan 头文件/glslc/loader 来自 conda-forge（不下载 LunarG SDK tarball）；`CMAKE_PREFIX_PATH=/opt/vulkan` 是 `find_package(SPIRV-Headers)` 的必需项，不能删
- Linux repair 目标为 `manylinux_2_31`（gcc-toolset-14 产物引用 GLIBCXX_3.4.25，超出 2_28 白名单），且 repair 的 `LD_LIBRARY_PATH` 不能包含 `/opt/vulkan/lib`（避免 auditwheel 解析到 conda 的新版 libstdc++）
- 发布新 wheel 后需同步更新 `requirements.txt` 的两个 URL 和两个 README 的平台说明

## 依赖

| 包 | 用途 |
|----|------|
| llama-cpp-python | llama.cpp Python 绑定（自编译 Vulkan wheel） |
| scipy | `gaussian_filter` 用于 BBox 遮罩羽化（在局部窗口上计算，非全图） |
| numpy | 图像数组操作 |
| pillow | 图像编解码、BBox 绘制 |
| tqdm | 终端进度条 |

## 已知问题

1. **单模型实例**: `LLAMA_CPP_STORAGE` 是全局单例，不支持同时加载多个模型
2. **核显不可选（有独显时）**: llama.cpp 的设备收集规则决定了有独显时核显无法通过 `main_gpu` 选中；如需强制核显推理，只能在进程启动前设置 `GGML_VK_VISIBLE_DEVICES` 环境变量（devices.py import 时即初始化 Vulkan，之后设置无效）
3. **import 时初始化 Vulkan**: 设备枚举在插件加载时同步执行（约几百 ms），属有意设计（UI 下拉框需要启动期确定设备列表）
