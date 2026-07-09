# ComfyUI-llama-cpp-vulkan

Run LLM/VLM models natively in ComfyUI based on llama.cpp with **Vulkan** GPU acceleration.

**[[📃中文版](./README_zh.md)]**

## Preview

![](./img/preview.jpg)

## Why Vulkan?

| Backend | AMD GPU | NVIDIA GPU | Intel GPU | Install Difficulty |
|---------|---------|------------|-----------|-------------------|
| CPU     | N/A     | N/A        | N/A       | None              |
| CUDA    | No      | Yes        | No        | Medium            |
| ROCm    | Partial | No         | No        | Hard              |
| **Vulkan** | **Yes** | **Yes** | **Yes** | **Easy**       |

Vulkan provides near-native GPU performance without requiring CUDA or ROCm toolkits. If your system can run games, it can run LLMs with Vulkan.

## Supported Models

### VLM (Vision-Language) Handlers

| Handler | Thinking Mode |
|---------|:------------:|
| Qwen3.5 / Qwen3.6 | Yes |
| Qwen3-VL | Yes |
| Qwen2.5-VL | - |
| Gemma3 / Gemma4 | - |
| GLM-4.6V / GLM-4.1V | Yes |
| LFM2-VL / LFM2.5-VL | - |
| MiniCPM-v4.5 / v4.6 | Yes |
| Step3-VL | - |

### Specialized Handlers

| Handler | Type |
|---------|------|
| DeepSeek-OCR | OCR |
| Granite-Docling | Document |
| PaddleOCR-VL-1.5 | OCR |
| MinerU2.5-Pro | Document |
| Qwen3-ASR | Speech |

> Any GGUF model without a specific handler will run in generic text mode.

## Installation

#### 1. Install the node:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan.git
```

#### 2. Install dependencies (includes pre-built Vulkan wheels):

```bash
pip install -r ComfyUI-llama-cpp-vulkan/requirements.txt
```

Pre-built Vulkan wheels are available for:
- Windows x86_64: Python 3.10 ~ 3.13
- Linux x86_64 (manylinux2014): Python 3.10 ~ 3.13

Download from [Releases](https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan/releases).

#### Alternative: Build from source

```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_VULKAN=1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

```bash
# Linux
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> Requires [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) when building from source.

#### 3. Download models:

- Place your `.gguf` model files in the `ComfyUI/models/LLM` folder.
- Custom paths via `extra_model_paths.yaml` are also supported.

  > If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
