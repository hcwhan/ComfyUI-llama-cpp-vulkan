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

## Installation

#### 1. Install the node:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan.git
```

#### 2. Install llama-cpp-python with Vulkan support:

**Pre-built wheel (recommended):**

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

**Or build from source:**

```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_VULKAN=1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

```bash
# Linux / macOS
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> Requires [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) to be installed on your system.

#### 3. Install other dependencies:

```bash
pip install -r ComfyUI-llama-cpp-vulkan/requirements.txt
```

#### 4. Download models:

- Place your `.gguf` model files in the `ComfyUI/models/LLM` folder.

  > If you need a VLM model to process image input, don't forget to download the `mmproj` weights.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
