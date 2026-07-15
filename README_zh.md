# ComfyUI-llama-cpp-vulkan

在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型, 使用 **Vulkan** 后端实现跨平台 GPU 加速.

**[[English](./README.md)]**

## 为什么选择 Vulkan?

| 后端 | AMD GPU | NVIDIA GPU | Intel GPU | 安装难度 |
|------|---------|------------|-----------|---------|
| CPU  | 不适用   | 不适用      | 不适用     | 无       |
| CUDA | 不支持   | 支持        | 不支持     | 中等     |
| ROCm | 部分支持 | 不支持      | 不支持     | 困难     |
| **Vulkan** | **支持** | **支持** | **支持** | **简单** |

Vulkan 无需 CUDA 或 ROCm 工具链即可实现接近原生的 GPU 加速性能. 只要你的系统能运行游戏, 就能用 Vulkan 加速 LLM 推理.

## 支持的模型

### VLM (视觉语言模型) Handlers

| Handler | 思维模式 |
|---------|:------:|
| Qwen3.6 / Qwen3.5 | 支持 |
| Qwen3-VL | 支持 |
| Qwen2.5-VL | - |
| GLM-4.6V | 支持 |
| GLM-4.1V | 恒开启 |
| Gemma4 | 支持 |
| Gemma3 | - |
| MiniCPM-v4.6 / v4.5 | 支持 |
| Step3-VL | 支持 |
| LFM2.5-VL / LFM2-VL | - |

> 思维模式由 vlm Model Loader 的 `thinking` 开关控制 (构造期参数, 切换后重新加载模型). 标注"支持"的 handler 开关可用; GLM-4.1V 是纯思考模型, 开关强制为开; 标注 "-" 的不支持思考, 开关强制为关 (前端自动置灰, 后端钳制兜底). Gemma4 E2B/E4B 即使关闭也会思考 (仅 31B/26BA4B 真正支持开关), 思考内容都会从输出中剥离. 下拉框按家族分组 (新版本在前), 完整列表以下拉框为准.

此外还提供早期模型的 handler (LLaVA-1.6 / 1.5, llama3-Vision-Alpha, nanoLLaVA, Moondream2, Obsidian, MiniCPM-v2.6).

### 专用 Handlers

| Handler | 类型 |
|---------|------|
| (ASR) Qwen3-ASR | 语音识别 |
| (OCR) DeepSeek-OCR | OCR 文字识别 |
| (OCR) PaddleOCR-VL-1.5 | OCR 文字识别 |
| (OCR) MinerU2.5-Pro | 文档解析 (基于 Qwen2.5-VL) |
| (OCR) Granite-Docling | 文档解析 |
| -Generic- | 兜底 Handler (渲染模型内置 chat template) |

> 没有专用 Handler 的 VLM 可以选择 `-Generic-` 兜底 Handler 以多模态方式运行; 纯文本 GGUF 模型无需 Handler, 用 `llm Model Loader` 以通用文本模式运行.

## 安装步骤

#### 1. 安装节点:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan.git
```

#### 2. 安装依赖 (包含预编译的 Vulkan wheel):

```bash
pip install -r ComfyUI-llama-cpp-vulkan/requirements.txt
```

> 无论通过何种方式安装节点 (git clone, ComfyUI-Manager 或 Comfy Registry), 此步骤都不可省略: `llama-cpp-python` 的 Vulkan wheel 只在 `requirements.txt` 中声明, 不在 Registry 元数据中.

预编译的 Vulkan wheel 支持以下平台:
- Windows x86_64: CPython 3.10+ (ABI3 通用 wheel)
- Linux x86_64 (manylinux_2_31, glibc >= 2.31): CPython 3.10+ (ABI3 通用 wheel)

每个平台只需一个 wheel 即可覆盖所有 Python 版本. 可从 [Releases](https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan/releases) 页面下载.

> 仅支持 requirements.txt 固定的预编译 wheel. 插件依赖该 Vulkan 构建 ([JamePeng 分支](https://github.com/JamePeng/llama-cpp-python)) 特有的 API, PyPI 官方包 (`pip install llama-cpp-python`) 缺少这些接口, 无法工作.

#### 3. 模型路径:

- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中 (`llm` 与 `LLM` 两个目录名均已注册, Linux 大小写敏感文件系统下任选其一).
- 也支持通过 `extra_model_paths.yaml` 配置自定义路径.

  > 在使用 VLM 模型进行图像推理之前, 请确保已经下载并选择了主模型对应的 `mmproj` 权重文件.
  > 加载器按文件名区分 mmproj 与主模型: 只有文件名含 `mmproj` (不区分大小写) 的文件才会出现在 mmproj 下拉框中, 重命名时请保留 `mmproj` 字样.

## 节点说明 (v2.1.0)

- **加载器**: `llm Model Loader` 加载纯文本 GGUF 模型, `vlm Model Loader` 加载视觉/音频模型 (mmproj 与 chat handler 必选). 两者输出类型完全独立: llm 只能连 `text Instruct`, vlm 只能连 `image / video / audio Instruct`.
- **推理节点**: 每个模态一个节点 - `text` (prompt 改写等), `image` (逐张或合并批量), `video` (输入为 IMAGE 帧批次, 均匀抽帧), `audio` (语音识别 / omni).
- **BBox 工具链**: `JSON to BBoxes` (解析检测 JSON 并画框), `BBoxes to SEGS` (兼容 Impact Pack), `BBoxes to MASK`, `BBoxes to BBox`.
- **实用工具**: `Parameters` (采样参数), `Unload Model`, `Parse JSON`, `Unpack Code Block`, `Split Instruct Output`, `System Prompt Preset` (Qwen-Image / Z-Image / Flux.2 / Wan 的中文提示词增强预设).

## 注意事项

- **音频输入 (语音识别)**: 将 ComfyUI 的 `AUDIO` 输出连接到 `audio Instruct` 节点, 模型用 `vlm Model Loader` 加载并选择支持音频的 handler (如 `(ASR) Qwen3-ASR`) 及其配套 mmproj. 音频以 16-bit 单声道 WAV 发送, 重采样由 llama.cpp 完成.
- **无状态推理**: 每次运行都是独立的一次性请求 (system prompt + 本次提问), 不保留任何跨运行的对话历史.
- **检测不到 GPU?** 用 ComfyUI 的 Python 运行 `python tools/check_devices.py`, 无需启动 ComfyUI 即可列出 GGML 后端枚举到的全部设备 (CPU/GPU/IGPU/ACCEL), 用于排查 Vulkan 驱动问题.

## 已知限制

- **单模型实例**: 插件维护一个全局模型槽位, 加载不同配置时自动卸载旧模型, 无法同时驻留两个模型.
- **多分片 GGUF (split shards) 不支持**: 显存折算只计所选文件的体积, 且下拉框中列出的非首分片选中会加载失败. 请先用 `llama-gguf-split --merge` 合并为单文件.
- **mmproj 不跟随显式选卡**: mtmd 只有 `use_gpu` 布尔开关 (上游限制), 多卡下显式选择非默认 GPU 时, 视觉编码器可能仍落在 mtmd 默认挑选的设备上.

## 致谢

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
