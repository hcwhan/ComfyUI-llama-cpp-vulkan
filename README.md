# ComfyUI-llama-cpp-vulkan

Run LLM/VLM models natively in ComfyUI based on llama.cpp with **Vulkan** GPU acceleration.

**[[中文版](./README_zh.md)]**

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
| Qwen3.6 / Qwen3.5 | Yes |
| Qwen3-VL | Yes |
| Qwen2.5-VL | - |
| GLM-4.6V | Yes |
| GLM-4.1V | Always on |
| Gemma4 | Yes |
| Gemma3 | - |
| MiniCPM-v4.6 / v4.5 | Yes |
| Step3-VL | Yes |
| LFM2.5-VL / LFM2-VL | - |

> Handlers marked "Yes" appear twice in the dropdown: the plain entry disables thinking, the `-Thinking` variant enables it. GLM-4.1V is a thinking-only model with no toggle. Gemma4 E2B/E4B think regardless of the toggle (only 31B/26BA4B honor it); the reasoning is stripped from the output either way. The dropdown groups handlers by family (newest first) and is the authoritative list.

Legacy handlers (LLaVA-1.6 / 1.5, llama3-Vision-Alpha, nanoLLaVA, Moondream2, Obsidian, MiniCPM-v2.6) are also available.

### Specialized Handlers

| Handler | Type |
|---------|------|
| Qwen3-ASR | Speech |
| DeepSeek-OCR | OCR |
| PaddleOCR-VL-1.5 | OCR |
| MinerU2.5-Pro | Document (based on Qwen2.5-VL) |
| Granite-Docling | Document |
| Generic-MTMD | Fallback (renders the model's built-in chat template) |

> VLMs without a dedicated handler can use the `Generic-MTMD` fallback handler. Text-only GGUF models need no handler at all and run in generic text mode via `llm Model Loader`.

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

> This step is required no matter how the node was installed (git clone, ComfyUI-Manager or Comfy Registry): the `llama-cpp-python` Vulkan wheel is only declared in `requirements.txt`, not in the registry metadata.

Pre-built Vulkan wheels are available for:
- Windows x86_64: CPython 3.10+ (ABI3 universal wheel)
- Linux x86_64 (manylinux_2_31, glibc >= 2.31): CPython 3.10+ (ABI3 universal wheel)

One wheel per platform covers all Python versions. Download from [Releases](https://github.com/hcwhan/ComfyUI-llama-cpp-vulkan/releases).

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
  > The loaders tell mmproj weights apart from main models by file name: only files whose name contains `mmproj` (case-insensitive) show up in the mmproj dropdown, so keep `mmproj` in the file name when renaming.

## Nodes (v2.0)

- **Loaders**: `llm Model Loader` for text-only GGUF models, `vlm Model Loader` for vision/audio models (mmproj + chat handler required). Their outputs are separate types: llm connects only to `text Instruct`, vlm connects only to `image / video / audio Instruct`.
- **Instruct**: one node per modality — `text` (prompt refining etc.), `image` (per-image or batched), `video` (IMAGE frame batch input, evenly sampled), `audio` (ASR / omni).
- **BBox toolchain**: `JSON to BBoxes` (parse detection JSON, draw boxes), `BBoxes to SEGS` (Impact Pack compatible), `BBoxes to MASK`, `BBoxes to BBox`.
- **Utilities**: `Parameters` (sampling config), `Unload Model`, `Parse JSON`, `Unpack Code Block`, `Split Instruct Output`, `System Prompt Preset` (Chinese prompt-enhancement presets for Qwen-Image / Z-Image / Flux.2 / Wan).

## Notes

- **Audio input (ASR)**: connect a ComfyUI `AUDIO` output to the `audio Instruct` node and load the model with `vlm Model Loader` using an audio-capable handler (e.g. Qwen3-ASR) with its matching mmproj. Audio is sent as 16-bit mono WAV; resampling is handled by llama.cpp.
- **Stateless inference**: every run is an independent one-shot request (system prompt + current prompt). No conversation history is kept between runs.
- **GPU not detected?** Run `python tools/check_devices.py` (with ComfyUI's Python) to list every device the GGML backend enumerates (CPU/GPU/IGPU/ACCEL) without starting ComfyUI — useful for diagnosing Vulkan driver issues.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
