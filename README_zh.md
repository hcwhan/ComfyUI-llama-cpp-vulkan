# ComfyUI-llama-cpp-vulkan

在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型, 使用 **Vulkan** 后端实现跨平台 GPU 加速。

**[[📃English](./README.md)]**

## 为什么选择 Vulkan?

| 后端 | AMD GPU | NVIDIA GPU | Intel GPU | 安装难度 |
|------|---------|------------|-----------|---------|
| CPU  | 不适用   | 不适用      | 不适用     | 无       |
| CUDA | 不支持   | 支持        | 不支持     | 中等     |
| ROCm | 部分支持 | 不支持      | 不支持     | 困难     |
| **Vulkan** | **支持** | **支持** | **支持** | **简单** |

Vulkan 无需 CUDA 或 ROCm 工具链即可实现接近原生的 GPU 加速性能。只要你的系统能运行游戏, 就能用 Vulkan 加速 LLM 推理。

## 支持的模型

### VLM (视觉语言模型) Handlers

| Handler | 思维模式 |
|---------|:------:|
| Qwen3.5 / Qwen3.6 | 支持 |
| Qwen3-VL | 支持 |
| Qwen2.5-VL | - |
| Gemma3 / Gemma4 | - |
| GLM-4.6V / GLM-4.1V | 支持 |
| LFM2-VL / LFM2.5-VL | - |
| MiniCPM-v4.5 / v4.6 | 支持 |
| Step3-VL | - |

此外还提供早期模型的 handler (LLaVA-1.5 / 1.6、Moondream2、nanoLLaVA、llama3-Vision-Alpha、MiniCPM-v2.6); 完整列表以 `vlm Model Loader` 下拉框为准。

### 专用 Handlers

| Handler | 类型 |
|---------|------|
| DeepSeek-OCR | OCR 文字识别 |
| Granite-Docling | 文档解析 |
| PaddleOCR-VL-1.5 | OCR 文字识别 |
| MinerU2.5-Pro | 文档解析 |
| Qwen3-ASR | 语音识别 |

> 没有专用 Handler 的 GGUF 模型会以通用文本模式运行。

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

预编译的 Vulkan wheel 支持以下平台:
- Windows x86_64: CPython 3.10+ (ABI3 通用 wheel)
- Linux x86_64 (manylinux_2_31, glibc >= 2.31): CPython 3.10+ (ABI3 通用 wheel)

每个平台只需一个 wheel 即可覆盖所有 Python 版本。可从 [Releases](https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan/releases) 页面下载。

#### 备选: 从源码编译

```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_VULKAN=1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

```bash
# Linux
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> 从源码编译需要先安装 [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)。

#### 3. 模型路径:

- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中。
- 也支持通过 `extra_model_paths.yaml` 配置自定义路径。

  > 在使用 VLM 模型进行图像推理之前, 请确保已经下载并选择了主模型对应的 `mmproj` 权重文件。

## 节点说明 (v2.0)

- **加载器**: `llm Model Loader` 加载纯文本 GGUF 模型, `vlm Model Loader` 加载视觉/音频模型 (mmproj 与 chat handler 必选)。两者输出类型完全独立: llm 只能连 `text Instruct`, vlm 只能连 `image / video / audio Instruct`。
- **推理节点**: 每个模态一个节点 — `text` (prompt 改写等)、`image` (逐张或合并批量)、`video` (输入为 IMAGE 帧批次, 均匀抽帧)、`audio` (语音识别 / omni)。
- **BBox 工具链**: `JSON to BBoxes` (解析检测 JSON 并画框)、`BBoxes to SEGS` (兼容 Impact Pack)、`BBoxes to MASK`、`BBoxes to BBox`。
- **实用工具**: `Parameters` (采样参数)、`Unload Model`、`Parse JSON`、`Unpack Code Block`、`Split Instruct Output`、`System Prompt Preset` (Qwen-Image / Z-Image / Flux.2 / Wan 的中文提示词增强预设)。

## 注意事项

- **音频输入 (语音识别)**: 将 ComfyUI 的 `AUDIO` 输出连接到 `audio Instruct` 节点, 模型用 `vlm Model Loader` 加载并选择支持音频的 handler (如 Qwen3-ASR) 及其配套 mmproj。音频以 16-bit 单声道 WAV 发送, 重采样由 llama.cpp 完成。
- **无状态推理**: 每次运行都是独立的一次性请求 (system prompt + 本次提问), 不保留任何跨运行的对话历史。
- **v1.x 旧工作流**: 原 `Model Loader` / `Instruct` 节点已被上述拆分节点取代, 加载旧工作流时需重建这些节点。
- **检测不到 GPU?** 用 ComfyUI 的 Python 运行 `python scripts/check_devices.py`, 无需启动 ComfyUI 即可列出 GGML 后端枚举到的全部设备 (CPU/GPU/IGPU/ACCEL), 用于排查 Vulkan 驱动问题。

## 致谢

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
