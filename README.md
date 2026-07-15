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

> Thinking is controlled by the `thinking` toggle on the VLM Model Loader (a construction-time parameter; changing it reloads the model). Handlers marked "Yes" honor the toggle; GLM-4.1V is a thinking-only model where the toggle is forced on; handlers marked "-" do not support thinking and the toggle is forced off (greyed out in the UI, clamped on the backend). Gemma4 E2B/E4B think even when disabled (only 31B/26BA4B honor the toggle); the reasoning is stripped from the output either way. The dropdown groups handlers by family (newest first) and is the authoritative list.

Legacy handlers (LLaVA-1.6 / 1.5, llama3-Vision-Alpha, nanoLLaVA, Moondream2, Obsidian, MiniCPM-v2.6) are also available.

### Specialized Handlers

| Handler | Type |
|---------|------|
| (ASR) Qwen3-ASR | Speech |
| (OCR) DeepSeek-OCR | OCR |
| (OCR) PaddleOCR-VL-1.5 | OCR |
| (OCR) MinerU2.5-Pro | Document (based on Qwen2.5-VL) |
| (OCR) Granite-Docling | Document |
| -Generic- | Fallback (renders the model's built-in chat template) |

> VLMs without a dedicated handler can use the `-Generic-` fallback handler. Text-only GGUF models need no handler at all and run in generic text mode via `LLM Model Loader`.

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

> Only the pre-built wheels pinned in `requirements.txt` are supported. The plugin relies on APIs specific to this Vulkan build ([JamePeng fork](https://github.com/JamePeng/llama-cpp-python)); the official PyPI package (`pip install llama-cpp-python`) lacks them and will not work.

#### 3. Download models:

- Place your `.gguf` model files in the `ComfyUI/models/llm` folder (both `llm` and `LLM` folder names are registered - on case-sensitive Linux filesystems either directory works).
- Custom paths via `extra_model_paths.yaml` are also supported.

  > If you need a VLM model to process image input, don't forget to download the `mmproj` weights.
  > The loaders tell mmproj weights apart from main models by file name: only files whose name contains `mmproj` (case-insensitive) show up in the mmproj dropdown, so keep `mmproj` in the file name when renaming.

## Nodes

- **Loaders**: `LLM Model Loader` for text-only GGUF models, `VLM Model Loader` for vision/audio models (mmproj + chat handler required). Their outputs are separate types: LLM connects only to `text Instruct`, VLM connects only to `image / video / audio Instruct`.
- **Instruct**: one node per modality - `text` (prompt refining etc.), `image` (per-image or batch), `video` (IMAGE frame batch input, evenly sampled), `audio` (ASR / Omni).
- **BBox toolchain**: `JSON to BBoxes` (parse detection JSON, draw boxes), `BBoxes to SEGS` (Impact Pack compatible), `BBoxes to MASK`, `BBoxes to BBox`.
- **Utilities**: `Parameters` (sampling config), `Unload Model`, `Parse JSON`, `Unpack Code Block`, `Split Instruct Output`, `System Prompt Preset` (Chinese prompt-enhancement presets for Qwen-Image / Z-Image / Flux.2 / Wan).

## Notes

- **Audio input (ASR)**: connect a ComfyUI `AUDIO` output to the `audio Instruct` node and load the model with `VLM Model Loader` using an audio-capable handler (e.g. `(ASR) Qwen3-ASR`) with its matching mmproj. Audio is sent as 16-bit mono WAV; resampling is handled by llama.cpp.
- **Stateless inference**: every run is an independent one-shot request (system prompt + current prompt). No conversation history is kept between runs.
- **GPU not detected?** Run `python tools/check_devices.py` (with ComfyUI's Python) to list every device the GGML backend enumerates (CPU/GPU/IGPU/ACCEL) without starting ComfyUI - useful for diagnosing Vulkan driver issues.

## Known Limitations

- **Single model instance**: the plugin keeps one global model slot. Loading a different configuration automatically unloads the previous model; two models cannot be resident at the same time.
- **Multi-shard GGUF (split shards) not supported**: VRAM estimation only counts the selected file, and non-first shards listed in the dropdown fail to load. Merge shards into a single file first with `llama-gguf-split --merge`.
- **mmproj does not follow explicit GPU selection**: mtmd only exposes a boolean `use_gpu` switch (upstream limitation), so on multi-GPU systems the vision encoder may stay on mtmd's default device when a non-default GPU is selected.
- **iGPU not selectable when a dGPU is present**: llama.cpp's device collection rule only reaches the iGPU when no discrete GPU exists. To force iGPU inference, set the `GGML_VK_VISIBLE_DEVICES` environment variable before the ComfyUI process starts - the plugin initializes Vulkan at import time, so setting it any later has no effect.
- **Vulkan initializes at plugin import**: device enumeration runs synchronously while the plugin loads (a few hundred ms, by design - the GPU dropdown needs the device list at startup). Keep this in mind when attributing ComfyUI startup time.

## Credits

- [llama-cpp-python](https://github.com/JamePeng/llama-cpp-python) @JamePeng
- [ComfyUI-llama-cpp](https://github.com/kijai/ComfyUI-llama-cpp) @kijai
