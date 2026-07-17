"""en-US language copy, the English source of all user-visible UI copy and console logs.

Structure conventions:
- display_names: node display names (NODE_DISPLAY_NAME_MAPPINGS).
- common: cross-group shared copy (used by nodes of multiple groups, single code source, one edit applies everywhere).
- nodes: organized by node group (matching the grouping of the src/nodes/ directory):
  model / instruct / bbox (BBox toolchain) / util.
  Within each group: common is the group-shared copy (used only by nodes of the group),
  other keys are per-node copy, grouped by description / tooltips / placeholders / errors.
- logs: console logs, grouped by source module (mostly one-to-one with code files; the bbox group spans multiple files, see its comment).

Layout conventions:
- Texts containing newlines use the parenthesized multi-literal form, one source line per UI display line
  (adjacent string literals are concatenated at compile time); pure single-line texts are single literals without parentheses.
- In the multi-literal form, every literal except the last must end with \\n
  (keeping source lines and display lines one-to-one, a missing \\n glues two display lines into one);
  the last literal must not end with \\n (UI text carries no trailing blank line).
- Copy in errors groups (key names ending with errors) must be single-line (error toasts do not render multi-line layout).
- The conventions above are locked by tests/test_i18n_format.py.

Placeholder conventions:
- {name} in error templates is a runtime str.format named placeholder, the name is fixed, surrounding text is editable.
- The !r in {item!r} fills the value in repr form, keep it together with the name.
- {{ and }} are escaped literal braces, rendered as { and }.
- {default} in parameters node tooltips is auto-filled by code from the widget default value.

Not in this file (not switched with language), respective homes:
- Dropdown option values (widget values serialized into workflows): Per-Image/Batch, Auto (GPU First) etc. in common_static.py; the chat handler list is the keys of the core/handlers.py registry.
- The node category (llama-cpp-vulkan) and the "======== Image N ========" prefix line (protocol strings matched by regex): common_static.py.
- Task presets and system prompt presets (names and contents): core/prompts.py and nodes/util/system_prompt_presets.py.
"""

LANG = {
    # ---- Node display names ----
    "display_names": {
        "llama_cpp_llm_model_loader": "llama.cpp LLM Model Loader",
        "llama_cpp_vlm_model_loader": "llama.cpp VLM Model Loader",
        "llama_cpp_parameters": "llama.cpp Parameters",
        "llama_cpp_unload_model": "llama.cpp Unload Model",
        "llama_cpp_text_instruct": "llama.cpp Text Instruct",
        "llama_cpp_image_instruct": "llama.cpp Image Instruct",
        "llama_cpp_video_instruct": "llama.cpp Video Instruct",
        "llama_cpp_audio_instruct": "llama.cpp Audio Instruct",
        "json_to_bboxes": "JSON to BBoxes",
        "bboxes_to_segs": "BBoxes to SEGS",
        "bboxes_to_mask": "BBoxes to MASK",
        "bboxes_to_bbox": "BBoxes to BBox",
        "parse_json_node": "Parse JSON",
        "remove_code_block": "Unpack Code Block",
        "split_instruct_output": "Split Instruct Output",
        "system_prompt_preset": "System Prompt Preset",
    },

    # ---- Cross-group shared copy ----
    "common": {
        # Shared model-loading errors (core/storage.py)
        "storage_errors": {
            "model_not_found": "Model '{model}' not found in the llm folder.",
            "unknown_chat_handler": 'Unknown chat handler: "{chat_handler}"',
            "handler_unavailable": 'Chat handler "{chat_handler}" is unavailable in this llama-cpp-python build (see startup warnings).',
            "mmproj_not_found": "mmproj '{mmproj}' not found in the llm folder.",
            "handler_required_for_mmproj": "Please select a matching chat handler for the vision model.",
            "mmproj_required_for_handler": 'Chat handler "{chat_handler}" requires a matching mmproj file.',
            "handler_init_failed": "Chat handler initialization failed. Check that the mmproj file matches the selected chat_handler, and that dependencies were installed from requirements.txt (pinned Vulkan wheel). {e}",
        },

        # GGUF parsing errors (core/gguf_layers.py)
        "gguf_errors": {
            "not_gguf": "Not a valid GGUF file!",
            "version_too_old": "GGUF v{version} is too old (v2+ required)",
            "unknown_value_type": "Unknown gguf metadata value type {vtype}",
            "array_count_implausible": "Implausible gguf metadata array length ({count}); file may be corrupted or misaligned",
            "string_length_implausible": "Implausible gguf metadata string length ({length}); file may be corrupted or misaligned",
        },

        # Shared JSON parsing error (parse_json in shared/text_utils.py)
        "errors": {
            "unable_to_load_json": "Unable to parse JSON data: {e}",
        },
    },

    "nodes": {
        # ================ model ================
        # llm_loader / vlm_loader / parameters / unload
        "model": {
            "common": {
                # Fields shared by both Model Loaders (node_loaders.py)
                "tooltips": {
                    "gpu_device": (
                        "Select the GPU device used for inference.\n"
                        "Auto = llama.cpp default behavior: discrete GPUs first, layer split across multiple discrete GPUs.\n"
                        "Selecting a specific device loads the whole model onto that single card.\n"
                        "The integrated GPU is selectable only when no discrete GPU is present."
                    ),
                    "ctx_size": (
                        "Context length limit, i.e. n_ctx of llama.cpp.\n"
                        "Prompt + generated tokens of a request cannot exceed this value.\n"
                        "KV cache grows linearly with it, oversizing wastes VRAM."
                    ),
                },
                # Same model-not-selected error for both Loaders (node_loaders.py)
                "errors": {
                    "model_not_selected": "Please select a gguf model file (placed in the llm folder).",
                },
            },

            # The vram_limit copy is maintained per Loader: the VLM side adds the mmproj budget deduction semantics
            "llm_loader": {
                "tooltips": {
                    "vram_limit": (
                        "VRAM usage limit in GB.\n"
                        "-1 = auto (llama.cpp fits layers to free VRAM);\n"
                        "0 = CPU-only inference;\n"
                        ">0 = offloads as many layers as fit by per-layer size (weights + KV cache), total usage never exceeds this value.\n"
                        "If the budget cannot fit even one model layer, the model stays on CPU (limit strictly honored);\n"
                        "with multiple discrete GPUs under Auto (layer split) the limit is the combined usage across all cards, not per card;\n"
                        "per-layer size is an estimate, actual usage may differ slightly."
                    ),
                },
            },

            "vlm_loader": {
                "tooltips": {
                    "thinking": (
                        "Enable the model's thinking (reasoning) mode.\n"
                        "Toggling reloads the whole model on the next run.\n"
                        "Only effective for handlers that support switching:\n"
                        "forced off for handlers without thinking support,\n"
                        "forced on for thinking-only models such as GLM-4.1V,\n"
                        "Gemma4 E2B/E4B still thinks in plain text when disabled.\n"
                        "Residual thinking content can be stripped by the strip_thinking switch of Instruct."
                    ),
                    "vram_limit": (
                        "VRAM usage limit in GB.\n"
                        "-1 = auto (llama.cpp fits layers to free VRAM);\n"
                        "0 = CPU-only inference (the mmproj stays on CPU too);\n"
                        ">0 = the mmproj size is deducted from the budget first, the rest offloads as many layers as fit by per-layer size (weights + KV cache), total usage never exceeds this value (limit strictly honored).\n"
                        "If the budget cannot fit the mmproj, both stay on CPU;\n"
                        "if less than one main-model layer fits after the deduction, the main model stays on CPU while the mmproj still goes to VRAM.\n"
                        "With multiple discrete GPUs under Auto (layer split) the limit is the combined usage across all cards, not per card;\n"
                        "per-layer size is an estimate, actual usage may differ slightly."
                    ),
                    "image_min_tokens": (
                        "Minimum tokens mmproj encodes per image, preserves encoding detail for low-resolution images.\n"
                        "0 = model default.\n"
                        "Only affects image/video inputs, audio is unaffected.\n"
                        "Changing it skews the Qwen2.5-VL mode coordinate mapping of JSON to BBoxes."
                    ),
                    "image_max_tokens": (
                        "Maximum tokens mmproj encodes per image, limits VRAM usage and encoding time for high-resolution images.\n"
                        "0 = model default.\n"
                        "Only affects image/video inputs, audio is unaffected.\n"
                        "Changing it skews the Qwen2.5-VL mode coordinate mapping of JSON to BBoxes."
                    ),
                },
                "errors": {
                    "mmproj_not_selected": "Please select the mmproj file paired with the model (placed in the llm folder); for text-only models use LLM Model Loader instead.",
                    "handler_not_selected": "Please select a chat handler matching the model.",
                    "image_token_range": "image_max_tokens ({image_max_tokens}) cannot be less than image_min_tokens ({image_min_tokens}).",
                },
            },

            # The trailing "default {default}." of each tooltip is auto-filled by code from the widget default value
            "parameters": {
                "tooltips": {
                    "max_gen_tokens": (
                        "Maximum tokens generated per run.\n"
                        "Effective limit = min(this value, ctx_size - prompt tokens),\n"
                        "output is silently truncated at the limit, no error raised.\n"
                        "0 = unlimited, default {default}."
                    ),
                    "top_k": (
                        "Keep only the K most probable candidate tokens for sampling.\n"
                        "0 = disabled, default {default}."
                    ),
                    "top_p": (
                        "Nucleus sampling: truncate candidates once cumulative probability reaches p, from high to low.\n"
                        "1.0 = disabled, default {default}."
                    ),
                    "min_p": (
                        "Relative probability floor: drop candidates below (top candidate probability x this value).\n"
                        "0.0 = disabled, default {default}."
                    ),
                    "typical_p": (
                        "Typical sampling: keep candidates whose information content is near the expected value, dropping overly surprising and overly bland tokens.\n"
                        "1.0 = disabled, default {default}."
                    ),
                    "temperature": (
                        "Sampling temperature: lower is more deterministic and conservative, higher is more divergent and random.\n"
                        "0.0 = greedy (always the most probable token), default {default}."
                    ),
                    "repeat_penalty": (
                        "Multiplicative penalty on tokens seen within the recent window.\n"
                        "1.0 = disabled, default {default}."
                    ),
                    "frequency_penalty": (
                        "Linear penalty accumulating per occurrence of a token, the more it appears the heavier the penalty, negative values reward repetition.\n"
                        "0.0 = disabled, default {default}."
                    ),
                    "present_penalty": (
                        "Penalizes a token once it has appeared at all, encouraging new content, negative values reward repetition.\n"
                        "0.0 = disabled, default {default}."
                    ),
                    "mirostat_mode": (
                        "Mirostat adaptive sampling: takes over sampling when enabled, top_k/top_p etc. are ignored.\n"
                        "0 = off, 1 = Mirostat, 2 = Mirostat 2.0, default {default}."
                    ),
                    "mirostat_eta": (
                        "Mirostat learning rate: controls how fast it converges to the target entropy, larger adjusts faster.\n"
                        "Only effective when mirostat_mode is on, default {default}."
                    ),
                    "mirostat_tau": (
                        "Mirostat target entropy: larger gives more diverse output, smaller is more focused and conservative.\n"
                        "Only effective when mirostat_mode is on, default {default}."
                    ),
                },
            },

            "unload": {
                "tooltips": {
                    "any": (
                        "Any-type passthrough port: data passes through unchanged,\n"
                        "chain it into the link where the model should be unloaded,\n"
                        "the model is unloaded to free VRAM when data flows through.\n"
                        "Note: runs only when the upstream output changes,\n"
                        "re-running an unchanged workflow does not unload again."
                    ),
                },
            },
        },

        # ================ instruct ================
        # The four Instruct nodes: text / image / video / audio
        "instruct": {
            # Shared fields and errors (core/instruct.py)
            "common": {
                "tooltips": {
                    "seed": (
                        "32-bit seed, capped at 0xFFFFFFFE,\n"
                        "avoiding llama.cpp's random-seed sentinel value 0xFFFFFFFF."
                    ),
                    "strip_thinking": "Remove thinking/reasoning blocks from the output.",
                    "force_offload": (
                        "Unload the model right after inference to free VRAM.\n"
                        "When off the model stays in VRAM, the next run skips reloading."
                    ),
                    "parameters": (
                        "Sampling parameter config.\n"
                        "When unconnected, behaves as if a Parameters node with all defaults were connected."
                    ),
                    "queue_handler": (
                        "Connect any upstream output to force this node to run after that upstream finishes,\n"
                        "used to order multiple Instruct nodes; the value itself takes no part in inference."
                    ),
                },
                "placeholders": {
                    "custom_prompt": (
                        "User prompt\n"
                        "\n"
                        "When the preset contains a placeholder, this content is required and fills the placeholder;\n"
                        "otherwise a non-empty value overrides the preset entirely, empty keeps the preset text."
                    ),
                    "system_prompt": (
                        "System prompt\n"
                        "\n"
                        "Sets the model's role and behavior constraints,\n"
                        "no system message is injected when empty."
                    ),
                },
                "errors": {
                    "preset_requires_custom_prompt": 'Preset "{preset_prompt}" contains a placeholder, please fill custom_prompt with the placeholder content.',
                    # Preset name mismatch (core/prompts.py, an old workflow feeding a renamed/removed preset name via a link)
                    "unknown_preset_prompt": 'Unknown preset: "{name}", please re-select from the dropdown (the workflow may reference a renamed or removed preset).',
                    # Raised only by text Instruct
                    "user_prompt_empty": "User prompt is empty: select a non-blank preset_prompt or fill custom_prompt.",
                    # Raised only by media Instructs; {kind} = Image / Video / Audio
                    "mmproj_not_configured": "{kind} input detected, but the loaded model is not configured with a mmproj module.",
                },
            },

            "text": {
                "tooltips": {
                    "allow_thinking": (
                        "Allow thinking models to output their reasoning process.\n"
                        "When off, a thinking block is force-closed as soon as it opens, skipping straight to the answer\n"
                        "(harmless for non-thinking models; residual empty thinking blocks can be stripped by strip_thinking)."
                    ),
                },
            },

            "image": {
                "tooltips": {
                    "mode": (
                        'Per-Image = each image is inferred separately for its own result; a single image is output directly, multiple images are joined into one output with "======== Image N ========" prefix lines (N starts at 1).\n'
                        "Batch = all images go into a single request (multiple images are scaled to max_size, a single image keeps its original resolution)."
                    ),
                    "increment_seed": (
                        "Per-Image mode only: when on, the N-th image uses seed+N-1 as its seed (the random-seed sentinel value is skipped),\n"
                        "so identical images can still produce different results; when off, all images reuse the same seed."
                    ),
                    "max_size": (
                        "Maximum edge length of input image resolution in Batch mode, downscaled proportionally when exceeded.\n"
                        "Only applies when sending multiple images, a single image keeps its original resolution."
                    ),
                },
            },

            "video": {
                "tooltips": {
                    "frames": "Video frames input as an Image frame batch (e.g. output of VHS Load Video or a video model's VAE Decode).",
                    "max_frames": (
                        "Upper limit of frames sampled uniformly from the input, the first and last frames are always taken;\n"
                        "all frames are taken when the input has no more than this value."
                    ),
                    "max_size": (
                        "Maximum edge length of sampled frame resolution, downscaled proportionally when exceeded.\n"
                        "Only applies when sending multiple frames, a single frame keeps its original resolution."
                    ),
                },
            },

            "audio": {
                "tooltips": {
                    "audio": (
                        "Audio clip for ASR/Omni models.\n"
                        "Requires an audio-capable mmproj.\n"
                        "Only the first clip of a multi-clip batch is processed."
                    ),
                },
            },
        },

        # ================ bbox (BBox toolchain) ================
        "bbox": {
            "json_to_bboxes": {
                "tooltips": {
                    "mode": (
                        "How model output coordinates are mapped to original-image pixel coordinates (both Qwen modes require the Image input):\n"
                        "Simple = passthrough (model output is already original-image pixel coordinates)\n"
                        "Qwen3-VL = 0-1000 normalized coordinates, restored by the original image size\n"
                        "Qwen2.5-VL = absolute coordinates in the model's internal resize space, restored to the original image automatically\n"
                        "  the mapping skews if the loader changed image_min/max_tokens;\n"
                        "  use together with the Per-Image mode of image Instruct:\n"
                        "    Batch mode with multiple images gets scaled by max_size and breaks the mapping,\n"
                        "    Batch mode with a single image is not scaled, the mapping stays exact"
                    ),
                    "label": (
                        "Keep only BBoxes with a matching label, empty keeps all.\n"
                        "(matching ignores case and surrounding whitespace, the text_content field is also recognized)"
                    ),
                },
                "errors": {
                    "image_required": "Qwen modes require the Image input",
                    # {i} is the index of the JSON segment (1-based, matching the Image N prefix line), {error} is the original parse error
                    "json_parse_failed": "Failed to parse JSON #{i}: {error}",
                    "not_a_list": 'Expected a JSON list of {{"bbox_2d": [...], "label": "..."}} objects, got: {type_name}',
                    # The next four live in bbox_utils.py: the first three are structure validation, unknown_mode guards the mode branch
                    "item_not_object": 'Expected list items to be objects like {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}}, got item: {item!r}',
                    "missing_bbox_2d": 'BBox item is missing a valid "bbox_2d": [x1, y1, x2, y2] field: {item!r}',
                    "coords_not_numeric": 'BBox "bbox_2d" coordinates must be numeric: {item!r}',
                    "unknown_mode": "Unknown coordinate mode: {mode}",
                },
            },

            "bboxes_to_segs": {
                "tooltips": {
                    "label": (
                        "Label written into each SEG, for downstream filtering/assignment by label.\n"
                        "(e.g. SEGS Filter of Impact Pack)"
                    ),
                    "confidence": "Confidence written into each SEG, for downstream threshold filtering.",
                    "dilation": (
                        "Pixels to expand the mask rectangle outward, directly enlarging the downstream repaint area.\n"
                        "(same dilation semantics as Impact Pack detectors and BBoxes to MASK)"
                    ),
                    "feather": (
                        "Gaussian feather sigma of the mask edge (pixels).\n"
                        "Only with crop_factor > 1 does the outward falloff of the mask edge have room inside crop_region;\n"
                        "with crop_factor = 1 the edge is clipped to a ~0.5 hard edge at the crop border, and dilation merely pushes it away from the original detection box."
                    ),
                    "crop_factor": (
                        "Scale of crop_region relative to the mask rectangle, giving the downstream Detailer repaint context.\n"
                        "(Impact Pack convention, 1.0 = no expansion)"
                    ),
                },
            },

            "bboxes_to_mask": {
                "tooltips": {
                    "dilation": (
                        "Pixels to expand the mask rectangle outward.\n"
                        "(same dilation semantics as BBoxes to SEGS)"
                    ),
                    "feather": "Gaussian feather sigma of the mask edge (pixels).",
                },
            },

            "bboxes_to_bbox": {
                "tooltips": {
                    "image_index": (
                        "Which image's BBox group to pick (0-based).\n"
                        "Group order matches the Image N numbering of the Per-Image multi-image output of image Instruct."
                    ),
                    "bbox_index": (
                        "BBox index within the image, negative counts from the end.\n"
                        "Set to 999 to return all BBoxes of that image."
                    ),
                },
                "errors": {
                    "image_index_out_of_range": "image_index {image_index} out of range: only {count} bbox group(s) available",
                    "bbox_index_out_of_range": "bbox_index {bbox_index} out of range: image {image_index} has only {count} bbox(es)",
                },
            },
        },

        # ================ util ================
        # parse_json / system_prompt_preset;
        # remove_code_block and split_instruct_output have no exclusive copy
        "util": {
            "parse_json": {
                "description": (
                    "Parse a JSON string and fetch a value by dotted key, outputting the same value as five types.\n"
                    "Conversion rules: string outputs valid JSON text for dict/list, str() result otherwise;\n"
                    "int/float fall back to 0 / 0.0 on conversion failure; boolean tests numbers for non-zero,\n"
                    'and for text only "true" (case-insensitive) is true.\n'
                    'When the key misses and default is empty (blank widget / unconnected), the five outputs are (None, "", 0, 0.0, False).'
                ),
                "tooltips": {
                    "key": (
                        "Fetch values level by level along a dotted path, e.g. a.b.c\n"
                        "Use numeric indices for array elements, e.g. items.0.label (negative counts from the end)"
                    ),
                },
                "errors": {
                    "key_empty": "key cannot be empty!",
                },
            },

            "system_prompt_preset": {
                "errors": {
                    "unknown_preset": 'Unknown preset: "{preset}", please re-select from the dropdown.',
                },
            },
        },
    },

    # ---- Console logs (grouped by source module, mostly one-to-one with code files, see the bbox group comment) ----
    # The fixed prefix "[llama-cpp-vulkan] " is a log filter tag, added at call sites, not in templates;
    # node execution logs carry an extra "[node name] " prefix (node_log_prefix in shared/logger.py,
    # node names are functional identifiers, not localized), so templates do not repeat the node name;
    # log levels (info/warning/debug) are code behavior, not in this file, special levels are noted in comments
    "logs": {
        # core/devices.py
        "devices": {
            "detection_failed": "GPU detection failed: {e}",
            # {summary} items are formatted as "name (description) [type]"
            "detected_devices": "Detected {count} GPU device(s): {summary}",
            "no_devices": "No GPU devices detected, running on CPU only",
            "device_not_selectable": "device '{gpu_device}' is not selectable, falling back to Auto",
            "no_backend": "No GPU backend detected, running on CPU only",
            "active_gpus_layer_split": "Active GPUs (layer split): {names}",
            "active_gpu": "Active GPU: {name} ({desc}) [{type}]",
        },

        # core/handlers.py
        "handlers": {
            # {missing} items are formatted as "display name (class name)"
            "handlers_unavailable": "chat handler(s) unavailable in this llama-cpp-python build: {missing}",
            "thinking_unsupported": 'handler "{label}" does not support thinking, the switch is treated as off',
            "thinking_forced": 'handler "{label}" is a thinking-only model, the switch is treated as on',
        },

        # core/storage.py
        "storage": {
            "vram_cannot_fit_mmproj": "vram_limit ({vram_limit} GB) cannot fit the mmproj file (~{mmproj_gb:.1f} GB), keeping the main model and mmproj on CPU to honor the budget",
            "vram_no_room_for_layer": "vram_limit ({vram_limit} GB) leaves no room for even one model layer (~{layer_size:.1f} GB/layer), keeping the main model on CPU to honor the budget",
            "kv_meta_fallback": "GGUF attention metadata incomplete; estimating KV cache from file size instead, vram_limit folding and VRAM estimates are coarser (KV underestimated for heavily quantized models)",
            # The next two are debug level, silent by default
            "llm_close_failed": "llm close failed: {e}",
            "handler_close_failed": "chat_handler close failed: {e}",
            "free_vram_request": "Asking ComfyUI to free {gb:.1f} GB of torch VRAM for model loading",
            "free_vram_failed": "failed to free torch VRAM before load: {e}",
            "preparing_mmproj": "Preparing mmproj: {mmproj}",
            "loading_model": "Loading model: {model}",
            "load_params": "n_gpu_layers = {n_gpu_layers}, n_layer = {n_layer}, main_gpu = {main_gpu}, split_mode = {split_mode}",
            "load_failed_retry": "model load failed ({e}), freeing torch VRAM and retrying once",
            "free_vram_retry_failed": "failed to free torch VRAM before retry: {free_err}",
            "load_finished": "Model loaded in {elapsed:.1f}s",
            "cpu_only": "CPU-only inference: no layers or mmproj offloaded to GPU",
            "mmproj_only_gpu": "all main model layers stay on CPU; only the mmproj (vision encoder) goes to VRAM (device picked by mtmd)",
            "unloaded": "Model resources unloaded",
            "cleanup_hook_applied": "Model cleanup hook applied!",
        },

        # core/instruct.py
        "instruct": {
            "request": 'Request: seed={seed}, preset="{preset}", custom_prompt {custom_chars} chars, system_prompt {system_chars} chars, strip_thinking={strip_thinking}, force_offload={force_offload}',
            "model_reused": "Reusing loaded model (config unchanged)",
            "interrupted": "Interrupt detected, aborting generation",
            # thinking/answer counts are estimates from re-tokenizing the answer text (wording carries "~"),
            # prompt tokens and total generated tokens come from the wheel's usage field; elapsed includes prompt prefill
            "generation_stats": "Prompt {prompt_tokens} tokens, generated {completion_tokens} tokens in {elapsed:.2f}s, {speed:.1f} tok/s",
            "generation_stats_thinking": "Prompt {prompt_tokens} tokens, generated {completion_tokens} tokens (~{thinking_tokens} thinking + ~{answer_tokens} answer) in {elapsed:.2f}s, {speed:.1f} tok/s",
            # debug level, silent by default
            "hybrid_reset": "hybrid/recurrent arch: KV cache fully reset after execution",
            # debug level, silent by default
            "think_probe_failed": "GGUF chat template probe render failed, treating as no pre-injected <think>: {e}",
        },

        # core/locale_sync.py
        "locale_sync": {
            # debug level, silent by default
            "frontend_locale_saved": "Frontend locale recorded: {locale} (effective next startup)",
        },

        # core/gguf_layers.py
        "gguf": {
            "parse_failed": "GGUF parse failed: {e}",
            "block_count_missing": "block_count not found in GGUF metadata",
        },

        # core/model_paths.py
        "model_paths": {
            "search_dirs": "GGUF model search directories: {dirs}",
        },

        # core/cqdm.py (description text of the tqdm terminal progress bar, not logger output)
        "cqdm": {
            "progress_desc": "Processing",
        },

        # shared/encoding.py
        "encoding": {
            "audio_batch_first_only": "AUDIO batch of {count} clips received; only the first clip is processed",
        },

        # nodes/model/node_loaders.py
        "loaders": {
            "llm_config": 'Config: model="{model}", ctx_size={n_ctx}, vram_limit={vram_limit}, gpu_device="{gpu_device}"',
            # thinking is the effective value after clamp_thinking
            "vlm_config": 'Config: model="{model}", mmproj="{mmproj}", chat_handler="{chat_handler}", thinking={thinking}, ctx_size={n_ctx}, vram_limit={vram_limit}, gpu_device="{gpu_device}", image_min/max_tokens={image_min_tokens}/{image_max_tokens}',
        },

        # nodes/model/node_parameters.py
        "parameters": {
            "sampling": "max_gen_tokens={max_gen_tokens}, top_k={top_k}, top_p={top_p}, min_p={min_p}, typical_p={typical_p}, temperature={temperature}, repeat_penalty={repeat_penalty}, frequency_penalty={frequency_penalty}, present_penalty={present_penalty}, mirostat_mode={mirostat_mode}, mirostat_eta={mirostat_eta}, mirostat_tau={mirostat_tau}",
        },

        # nodes/model/node_unload.py
        "unload": {
            "unloading": "Unloading llama model...",
        },

        # nodes/instruct/text/node_instruct.py
        "text_instruct": {
            "allow_thinking": "allow_thinking={allow_thinking}, mapped to reasoning_budget={reasoning_budget}, reasoning_start_in_prompt={reasoning_start_in_prompt}",
        },

        # nodes/instruct/media/image/node_instruct.py
        "image_instruct": {
            "each_mode": "Per-Image mode: {count} image(s), one request each, increment_seed={increment_seed}",
            "batch_mode": "Batch mode: {count} image(s) merged into a single message, one request, max_size={max_size}",
        },

        # nodes/instruct/media/video/node_instruct.py
        "video_instruct": {
            "sampling": "Frame sampling: {total} input frames, {sampled} sampled, max_size={max_size}",
        },

        # nodes/instruct/media/audio/node_instruct.py
        "audio_instruct": {
            "input": "Audio duration {duration:.1f}s, sample rate {sample_rate} Hz",
        },

        # the node files under nodes/bbox/ + bbox_utils.py
        "bbox": {
            # {detail} takes one of the two detail_* variants below
            "json_frame_mismatch": "{json_count} JSON result(s) but {frame_count} image frame(s); pairing by index, {detail}",
            "detail_extra_json": "extra JSON entries reuse the last frame, appended to image_list as single-frame batches",
            "detail_extra_frames": "unpaired trailing frames are passed through without boxes",
            "draw_failed_json": "Error drawing bboxes for JSON #{i}: {e}",
            "segs_batch_first_frame": "Received a batch of {batch_size} images; cropped images are taken from the first frame only",
            # Same text in both the SEGS and MASK paths
            "bbox_out_of_bounds": "Skipping bbox outside image bounds: {bbox}",
            "bbox_empty_area": "Skipping bbox with empty area: {bbox}",
            "no_cjk_font": "No CJK font found, bbox labels may render as boxes",
            "bbox_draw_failed": "Skipping bbox that failed to draw ({label!r}: ({x0}, {y0}, {x1}, {y1})): {e}",
            "bbox_invalid_item": "Skipping invalid bbox item: {bbox}",
            "bbox_non_numeric": "Skipping bbox with non-numeric coordinates: {bbox}",
            # per-node result summaries
            "json_to_bboxes_summary": '{json_count} JSON segment(s) parsed into {bbox_count} bbox(es) (mode={mode}, label filter="{label}")',
            "segs_summary": "{bbox_count} bbox(es) produced {seg_count} SEG(s) (dilation={dilation}, feather={feather}, crop_factor={crop_factor})",
            "mask_summary": "{bbox_count} bbox(es) merged into one mask covering {coverage:.1f}% of pixels (dilation={dilation}, feather={feather})",
            "bbox_selected": "Selected image_index={image_index}, bbox_index={bbox_index} -> {bbox}",
            "bbox_selected_all": "Whole group selected at image_index={image_index}, {count} bbox(es)",
        },

        # the node files under nodes/util/
        "util": {
            "parse_json": 'key="{key}" -> {type_name}, string output {chars} chars',
            "remove_code_block": "{before} chars in -> {after} chars out",
            "split_output": "Split into {count} segment(s)",
            "system_prompt": 'Using "{preset}", {chars} chars',
        },
    },
}
