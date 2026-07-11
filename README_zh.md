# ComfyUI-llama-cpp-vulkan

在 ComfyUI 中基于 llama.cpp 框架原生运行 LLM & VLM 模型, 使用 **Vulkan** 后端实现跨平台 GPU 加速。

**[[📃English](./README.md)]**

## 预览

![](./img/preview.jpg)

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

## 注意事项

- **音频输入 (语音识别)**: 将 ComfyUI 的 `AUDIO` 输出连接到 Instruct 节点的可选 `audio` 输入, 并选择支持音频的 handler (如 Qwen3-ASR) 及其配套 mmproj。音频以 16-bit 单声道 WAV 发送, 重采样由 llama.cpp 完成。
- **`save_states` 多轮对话与图片/音频**: 为节省内存, 保存会话历史时图片会被替换为 1x1 占位图 (音频替换为静音占位符)。后续轮次中模型无法"回看"之前的媒体内容, 针对历史图片的追问可能得到不准确的回答。如需围绕同一张图片多轮追问, 请在每轮都重新输入该图片。

## 致谢

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
