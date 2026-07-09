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

## 安装步骤

#### 1. 安装节点:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan.git
```

#### 2. 安装 Vulkan 版 llama-cpp-python:

**预编译包 (推荐):**

```bash
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan
```

**或从源码编译:**

```powershell
# Windows PowerShell
$env:CMAKE_ARGS="-DGGML_VULKAN=1"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

```bash
# Linux / macOS
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> 需要先安装 [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)。

#### 3. 安装其他依赖:

```bash
pip install -r ComfyUI-llama-cpp-vulkan/requirements.txt
```

#### 4. 模型路径:

- 请将下载的 `.gguf` 模型放置在 `ComfyUI/models/LLM` 目录中。

  > 在使用 VLM 模型进行图像推理之前, 请确保已经下载并选择了主模型对应的 `mmproj` 权重文件。

## 致谢

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
