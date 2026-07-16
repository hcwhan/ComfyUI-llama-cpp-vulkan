"""zh-CN 语言文案, UI 上用户可见全部文案和控制台日志的中文来源.

结构约定:
- display_names: 节点显示名 (NODE_DISPLAY_NAME_MAPPINGS).
- common: 跨组共享文案 (多个分组的节点都会用到, 代码同源, 改一处全生效).
- nodes: 按节点分组组织 (与 src/nodes/ 目录的分组一致):
  model (模型) / instruct (推理) / bbox (BBox 工具链) / util (工具).
  每组内: common 为组内共享文案 (仅本组节点用到),
  其余 key 为各节点专属文案, 按 description / tooltips / placeholders / errors 分组.
- logs: 控制台日志, 按来源模块分组 (多数与代码文件一一对应, bbox 组横跨多个文件, 见其注释).

排版约定:
- 含换行的文本用括号包裹的多字面量形态, 每个 UI 显示行对应一个源码行
  (相邻字符串字面量编译期自动拼接); 纯单行文本直接写单字面量, 不加括号.
- 多字面量形态中, 除最后一个外, 每个字面量必须以 \\n 结尾
  (保证源码行与显示行一一对应, 漏写 \\n 会把两个显示行粘连成一行);
  最后一个字面量不能以 \\n 结尾 (UI 文本不带尾随空行).
- errors 类分组 (键名以 errors 结尾) 的文案必须是单行 (报错弹窗不展示多行排版).
- 以上约定由 tests/test_i18n_format.py 锁定.

占位符约定:
- 报错模板中的 {name} 为运行时 str.format 具名占位符, 名称不可改, 周围文字可改.
- {item!r} 的 !r 表示以 repr 形态填充, 需连同保留.
- {{ 与 }} 是转义后的字面花括号, 渲染为 { 与 }.
- parameters 节点 tooltip 中的 {default} 由代码按 widget 默认值自动填充.

不在本文件 (不随语言切换), 各自归属:
- 下拉框选项值 (会序列化进工作流的 widget 值): Per-Image/Batch, Auto (GPU First) 等在 common_static.py; chat handler 名单即 core/handlers.py 注册表的 key.
- 节点分类名 category (llama-cpp-vulkan) 与 "======== Image N ========" 前缀行 (被正则匹配的协议串): common_static.py.
- 任务预设与系统提示词预设 (名称与内容): core/prompts.py 与 nodes/util/system_prompt_presets.py.
"""

LANG = {
    # ---- 节点显示名 ----
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

    # ---- 跨组共享文案 ----
    "common": {
        # 模型加载期共用报错 (core/storage.py)
        "storage_errors": {
            "model_not_found": "在 llm 目录中找不到模型 '{model}'.",
            "unknown_chat_handler": '未知的 chat handler: "{chat_handler}"',
            "handler_unavailable": 'chat handler "{chat_handler}" 在当前 llama-cpp-python 构建中不可用 (见启动日志 warning).',
            "mmproj_not_found": "在 llm 目录中找不到 mmproj '{mmproj}'.",
            "handler_required_for_mmproj": "请为视觉模型选择配套的 chat handler.",
            "mmproj_required_for_handler": 'chat handler "{chat_handler}" 需要配套的 mmproj 文件.',
            "handler_init_failed": "chat handler 初始化失败. 请检查 mmproj 文件与所选的 chat_handler 是否匹配, 以及依赖是否已按 requirements.txt 安装 (固定版本的 Vulkan wheel). {e}",
        },

        # GGUF 文件解析报错 (core/gguf_layers.py)
        "gguf_errors": {
            "not_gguf": "不是有效的 GGUF 文件!",
            "version_too_old": "GGUF v{version} 版本太低 (需要 v2 及以上)",
            "unknown_value_type": "未知的 gguf 元数据类型 {vtype}",
            "array_count_implausible": "gguf 元数据数组长度异常 ({count}), 文件可能损坏或错位",
            "string_length_implausible": "gguf 元数据字符串长度异常 ({length}), 文件可能损坏或错位",
        },

        # JSON 解析共用报错 (shared/text_utils.py 的 parse_json)
        "errors": {
            "unable_to_load_json": "无法解析 JSON 数据: {e}",
        },
    },

    "nodes": {
        # ================ model (模型) ================
        # llm_loader / vlm_loader / parameters / unload
        "model": {
            "common": {
                # 两个 Model Loader 共用字段 (node_loaders.py)
                "tooltips": {
                    "gpu_device": (
                        "选择推理使用的 GPU 设备.\n"
                        "Auto = llama.cpp 默认行为: 独显优先, 多独显时按层切分.\n"
                        "显式选择某设备时, 整个模型加载到该单卡.\n"
                        "仅当系统没有独显时, 核显才可选."
                    ),
                    "ctx_size": (
                        "上下文长度上限, 即 llama.cpp 的 n_ctx.\n"
                        "请求的 prompt + 生成 token 总量不能超过此值.\n"
                        "KV cache 随此值线性增长, 设置过大会浪费显存."
                    ),
                },
                # 两个 Loader 的未选模型报错同文 (node_loaders.py)
                "errors": {
                    "model_not_selected": "请选择 gguf 模型文件 (需放在 llm 目录).",
                },
            },

            # vram_limit 文案两个 Loader 分开维护: VLM 侧多 mmproj 预算扣除语义
            "llm_loader": {
                "tooltips": {
                    "vram_limit": (
                        "显存占用上限, 单位 GB.\n"
                        "-1 = 自动 (llama.cpp 按空闲显存适配层数);\n"
                        "0 = 纯 CPU 推理;\n"
                        ">0 = 按每层体积 (权重 + KV cache) 折算可上 GPU 的层数, 总占用不超过该值.\n"
                        "预算不足模型 1 层时模型留在 CPU (严格守上限);\n"
                        "多独显 Auto (按层切分) 下为全部卡占用的合计上限, 而非单卡上限;\n"
                        "层体积为估算值, 实际占用可能略有偏差."
                    ),
                },
            },

            "vlm_loader": {
                "tooltips": {
                    "thinking": (
                        "开启模型的思考 (reasoning) 模式.\n"
                        "切换后下次执行会整体重新加载模型.\n"
                        "仅对支持切换的 handler 生效:\n"
                        "不支持思考的强制为关,\n"
                        "GLM-4.1V 等纯思考模型强制为开,\n"
                        "Gemma4 E2B/E4B 关闭后仍会以纯文本形式思考.\n"
                        "残留的思考内容可由 Instruct 的 strip_thinking 开关剥离."
                    ),
                    "vram_limit": (
                        "显存占用上限, 单位 GB.\n"
                        "-1 = 自动 (llama.cpp 按空闲显存适配层数);\n"
                        "0 = 纯 CPU 推理 (mmproj 也留在 CPU);\n"
                        ">0 = mmproj 体积先从预算中扣除, 余量按每层体积 (权重 + KV cache) 折算可上 GPU 的层数, 总占用不超过该值 (严格守上限).\n"
                        "预算装不下 mmproj 时两者全留 CPU;\n"
                        "扣除后不足主模型 1 层时主模型留在 CPU, mmproj 仍进显存.\n"
                        "多独显 Auto (按层切分) 下为全部卡占用的合计上限, 而非单卡上限;\n"
                        "层体积为估算值, 实际占用可能略有偏差."
                    ),
                    "image_min_tokens": (
                        "mmproj 编码每张图片的最小 token 数, 可保证低分辨率图片的编码精细度.\n"
                        "0 = 使用模型默认.\n"
                        "仅对图像/视频输入生效, 音频不受影响.\n"
                        "修改后 JSON to BBoxes 的 Qwen2.5-VL 模式坐标换算会有偏差."
                    ),
                    "image_max_tokens": (
                        "mmproj 编码每张图片的最大 token 数, 可限制高分辨率图片的显存占用与编码耗时.\n"
                        "0 = 使用模型默认.\n"
                        "仅对图像/视频输入生效, 音频不受影响.\n"
                        "修改后 JSON to BBoxes 的 Qwen2.5-VL 模式坐标换算会有偏差."
                    ),
                },
                "errors": {
                    "mmproj_not_selected": "请选择与模型配对的 mmproj 文件 (需放在 llm 目录); 纯文本模型请改用 LLM Model Loader.",
                    "handler_not_selected": "请选择与模型匹配的 chat handler.",
                    "image_token_range": "image_max_tokens ({image_max_tokens}) 不能小于 image_min_tokens ({image_min_tokens}).",
                },
            },

            # 各 tooltip 末行的 "默认 {default}." 由代码按 widget 默认值填充
            "parameters": {
                "tooltips": {
                    "max_gen_tokens": (
                        "单次生成的 token 数上限.\n"
                        "实际上限 = min(该值, ctx_size - prompt tokens),\n"
                        "达到上限时输出被静默截断, 不报错.\n"
                        "0 = 不限制, 默认 {default}."
                    ),
                    "top_k": (
                        "只保留概率最高的 K 个候选 token 再采样.\n"
                        "0 = 禁用, 默认 {default}."
                    ),
                    "top_p": (
                        "核采样: 按概率从高到低累计到 p 截断候选集.\n"
                        "1.0 = 禁用, 默认 {default}."
                    ),
                    "min_p": (
                        "相对概率下限: 剔除概率低于 (最高候选概率 x 该值) 的候选.\n"
                        "0.0 = 禁用, 默认 {default}."
                    ),
                    "typical_p": (
                        "典型性采样: 保留信息量接近期望值的候选, 剔除过于意外与过于平庸的 token.\n"
                        "1.0 = 禁用, 默认 {default}."
                    ),
                    "temperature": (
                        "采样温度: 越低输出越确定保守, 越高越发散随机.\n"
                        "0.0 = 贪心 (恒取最高概率), 默认 {default}."
                    ),
                    "repeat_penalty": (
                        "对最近窗口内出现过的 token 乘惩罚系数.\n"
                        "1.0 = 禁用, 默认 {default}."
                    ),
                    "frequency_penalty": (
                        "按 token 已出现次数线性累加惩罚, 出现越多惩罚越重, 负值奖励重复.\n"
                        "0.0 = 禁用, 默认 {default}."
                    ),
                    "present_penalty": (
                        "token 只要出现过就惩罚一次, 鼓励引入新内容, 负值奖励重复.\n"
                        "0.0 = 禁用, 默认 {default}."
                    ),
                    "mirostat_mode": (
                        "Mirostat 自适应采样: 开启后接管采样, top_k/top_p 等失效.\n"
                        "0 = 关闭, 1 = Mirostat, 2 = Mirostat 2.0, 默认 {default}."
                    ),
                    "mirostat_eta": (
                        "Mirostat 学习率: 控制向目标熵收敛的速度, 越大调整越快.\n"
                        "仅 mirostat_mode 开启时生效, 默认 {default}."
                    ),
                    "mirostat_tau": (
                        "Mirostat 目标熵: 越大输出越多样, 越小越集中保守.\n"
                        "仅 mirostat_mode 开启时生效, 默认 {default}."
                    ),
                },
            },

            "unload": {
                "tooltips": {
                    "any": (
                        "any 透传端口: 数据原样通过,\n"
                        "串接在需要卸载模型的连线上,\n"
                        "流经该节点时卸载模型释放显存.\n"
                        "注意: 仅在上游输出变化时执行,\n"
                        "重跑未改动的工作流不会执行卸载."
                    ),
                },
            },
        },

        # ================ instruct (推理) ================
        # text / image / video / audio 四个 Instruct 节点
        "instruct": {
            # 共用字段与报错 (core/instruct.py)
            "common": {
                "tooltips": {
                    "seed": (
                        "32 位种子, 上限 0xFFFFFFFE,\n"
                        "避开 llama.cpp 的随机种子哨兵值 0xFFFFFFFF."
                    ),
                    "strip_thinking": "移除输出中的思考/推理块.",
                    "force_offload": (
                        "推理结束后立即卸载模型, 释放显存.\n"
                        "关闭时模型常驻显存, 下次执行免重新加载."
                    ),
                    "parameters": (
                        "采样参数配置.\n"
                        "不连接时等同于连接一个全默认值的 Parameters 节点."
                    ),
                    "queue_handler": (
                        "接入上游任意输出, 强制本节点在该上游完成后才执行,\n"
                        "用于控制多个 Instruct 节点的先后顺序; 值本身不参与推理."
                    ),
                },
                "placeholders": {
                    "custom_prompt": (
                        "用户提示词\n"
                        "\n"
                        "预设含占位符时, 此内容为必填, 用于填充占位符;\n"
                        "否则该值非空时整体覆盖预设, 留空则使用预设原文."
                    ),
                    "system_prompt": (
                        "系统提示词\n"
                        "\n"
                        "用于设定模型的角色与行为约束,\n"
                        "留空时不注入 system 消息."
                    ),
                },
                "errors": {
                    "preset_requires_custom_prompt": '预设 "{preset_prompt}" 含占位符, 请在 custom_prompt 中填写占位内容.',
                    # 预设名失配 (core/prompts.py, 旧工作流经连线传入已改名/删除的预设名)
                    "unknown_preset_prompt": '未知的预设: "{name}", 请在下拉框中重新选择 (工作流可能引用了已改名或删除的预设).',
                    # 仅 text Instruct 触发
                    "user_prompt_empty": "用户提示词为空: 请选择非空白的 preset_prompt, 或填写 custom_prompt.",
                    # 仅媒体 Instruct 触发; {kind} = Image / Video / Audio
                    "mmproj_not_configured": "检测到 {kind} 输入, 但当前加载的模型未配置 mmproj 模块.",
                },
            },

            "text": {
                "tooltips": {
                    "allow_thinking": (
                        "允许 Thinking 模型输出思考过程.\n"
                        "关闭时思考块一开启即被强制闭合, 跳过思考直接生成正文\n"
                        "(对非思考模型无副作用; 残留的空思考块可由 strip_thinking 剥离)."
                    ),
                },
            },

            "image": {
                "tooltips": {
                    "mode": (
                        'Per-Image = 逐张图片单独推理, 各得一条结果, 单图时直接输出, 多图时以 "======== Image N ========" (N 从 1 起) 作为前缀行然后拼接为单条输出.\n'
                        "Batch = 全部图片并入单次请求 (多图时缩放到 max_size, 单图保持原分辨率)."
                    ),
                    "increment_seed": (
                        "仅 Per-Image 模式生效: 开启后第 N 张图以 seed+N-1 为种子 (自动避开随机哨兵值),\n"
                        "使内容相同的图片也能得到不同结果; 关闭时全部图片复用同一 seed."
                    ),
                    "max_size": (
                        "Batch 模式下输入图片分辨率的最大边长, 超出时等比缩小.\n"
                        "仅在发送多张图片时生效, 单张图片保持原分辨率."
                    ),
                },
            },

            "video": {
                "tooltips": {
                    "frames": "以 Image 帧批次形式输入的视频帧 (如 VHS Load Video 或视频模型 VAE Decode 的输出).",
                    "max_frames": (
                        "从输入帧中均匀抽取的帧数上限, 首尾帧恒被抽取;\n"
                        "输入帧数不超过该值时全部抽取."
                    ),
                    "max_size": (
                        "采样帧分辨率的最大边长, 超出时等比缩小.\n"
                        "仅在发送多帧时生效, 单帧保持原分辨率."
                    ),
                },
            },

            "audio": {
                "tooltips": {
                    "audio": (
                        "供 ASR/Omni 模型使用的音频片段.\n"
                        "需要支持音频的 mmproj.\n"
                        "多段音频批次仅处理第一段."
                    ),
                },
            },
        },

        # ================ bbox (BBox 工具链) ================
        "bbox": {
            "json_to_bboxes": {
                "tooltips": {
                    "mode": (
                        "将模型输出坐标换算为原图像素坐标的方式 (两种 Qwen 模式均需连接 Image):\n"
                        "Simple = 原样透传 (模型输出即为原图像素坐标)\n"
                        "Qwen3-VL = 0-1000 归一化坐标, 按原图尺寸还原\n"
                        "Qwen2.5-VL = 模型内部 resize 空间的绝对坐标, 自动还原到原图\n"
                        "  loader 修改过 image_min/max_tokens 时换算会有偏差;\n"
                        "  需配合 image Instruct 的 Per-Image 模式使用:\n"
                        "    Batch 模式多图时会被 max_size 缩放导致换算失真,\n"
                        "    Batch 模式单图时不缩放, 换算仍精确"
                    ),
                    "label": (
                        "只保留 label 匹配的 BBox, 留空保留全部.\n"
                        "(匹配忽略大小写与首尾空格, 兼容 text_content 字段)"
                    ),
                },
                "errors": {
                    "image_required": "Qwen 模式需要连接 Image 输入",
                    # {i} 为该段 JSON 的序号 (从 1 起, 与前缀行 Image N 对齐), {error} 为原始解析错误
                    "json_parse_failed": "JSON #{i} 解析失败: {error}",
                    "not_a_list": '期望 JSON 为由 {{"bbox_2d": [...], "label": "..."}} 对象组成的列表, 实际为: {type_name}',
                    # 以下四条位于 bbox_utils.py: 前三条为结构校验, unknown_mode 为 mode 分支防御
                    "item_not_object": '期望列表项为形如 {{"bbox_2d": [x1, y1, x2, y2], "label": "..."}} 的对象, 实际列表项为: {item!r}',
                    "missing_bbox_2d": 'BBox 项缺少有效的 "bbox_2d": [x1, y1, x2, y2] 字段: {item!r}',
                    "coords_not_numeric": 'BBox 的 "bbox_2d" 坐标必须为数字: {item!r}',
                    "unknown_mode": "未知的坐标换算模式: {mode}",
                },
            },

            "bboxes_to_segs": {
                "tooltips": {
                    "label": (
                        "写入每个 SEG 的 label, 供下游按 label 过滤/赋值.\n"
                        "(如 Impact Pack 的 SEGS Filter)"
                    ),
                    "confidence": "写入每个 SEG 的置信度, 供下游按阈值过滤.",
                    "dilation": (
                        "掩码矩形向外扩张的像素数, 直接扩大下游的重绘区域.\n"
                        "(与 Impact Pack 检测器及 BBoxes to MASK 的 dilation 语义一致)"
                    ),
                    "feather": (
                        "掩码边缘高斯羽化的 sigma (像素).\n"
                        "crop_factor > 1 时掩码边缘的外侧衰减才有空间落入 crop_region;\n"
                        "crop_factor = 1 时边缘在 crop 边界被裁成 ~0.5 硬边, dilation 仅把硬边推离原始检测框."
                    ),
                    "crop_factor": (
                        "crop_region 相对掩码矩形的放大倍数, 为下游 Detailer 提供重绘上下文.\n"
                        "(Impact Pack 惯例, 1.0 = 不外扩)"
                    ),
                },
            },

            "bboxes_to_mask": {
                "tooltips": {
                    "dilation": (
                        "掩码矩形向外扩张的像素数.\n"
                        "(与 BBoxes to SEGS 的 dilation 语义一致)"
                    ),
                    "feather": "掩码边缘高斯羽化的 sigma (像素).",
                },
            },

            "bboxes_to_bbox": {
                "tooltips": {
                    "image_index": (
                        "选取第几张图的 BBox 组 (从 0 起).\n"
                        "组序与 image Instruct Per-Image 多图输出的 Image N 序号对齐."
                    ),
                    "bbox_index": (
                        "图内 BBox 索引, 负数从尾部取.\n"
                        "设为 999 时返回该图全部 BBox."
                    ),
                },
                "errors": {
                    "image_index_out_of_range": "image_index {image_index} 越界: 仅有 {count} 组 BBox",
                    "bbox_index_out_of_range": "bbox_index {bbox_index} 越界: 图 {image_index} 仅有 {count} 个 BBox",
                },
            },
        },

        # ================ util (工具) ================
        # parse_json / system_prompt_preset;
        # remove_code_block 与 split_instruct_output 无专属文案
        "util": {
            "parse_json": {
                "description": (
                    "解析 JSON 字符串并按点分 key 取值, 同一个值以五种类型输出.\n"
                    "转换规则: string 遇 dict/list 输出合法 JSON 文本, 其余为 str() 结果;\n"
                    "int/float 转换失败时回退 0 / 0.0; boolean 对数字按非零判定,\n"
                    '对文本仅 "true" (忽略大小写) 为真.\n'
                    'key 未命中且 default 为空 (未填写/未连线) 时, 五个输出为 (None, "", 0, 0.0, False).'
                ),
                "tooltips": {
                    "key": (
                        "按点分路径逐层取值, 如 a.b.c\n"
                        "数组元素用数字下标, 如 items.0.label (负数从尾部倒数)"
                    ),
                },
                "errors": {
                    "key_empty": "key 不能为空!",
                },
            },

            "system_prompt_preset": {
                "errors": {
                    "unknown_preset": '未知的预设: "{preset}", 请在下拉框中重新选择.',
                },
            },
        },
    },

    # ---- 控制台日志 (按来源模块分组, 多数与代码文件一一对应, bbox 组见其注释) ----
    # 固定前缀 "[llama-cpp-vulkan] " 是日志过滤标签, 不进模板, 由调用处添加;
    # 日志级别 (info/warning/debug) 属代码行为, 不在本文件, 特殊级别以注释标注
    "logs": {
        # core/devices.py
        "devices": {
            "detection_failed": "GPU 检测失败: {e}",
            # {summary} 列表项格式为 "名称 (描述) [类型]"
            "detected_devices": "检测到 {count} 个 GPU 设备: {summary}",
            "no_devices": "未检测到 GPU 设备, 仅使用 CPU 运行",
            "device_not_selectable": "设备 '{gpu_device}' 不可选, 回退为 Auto",
            "no_backend": "未检测到 GPU 后端, 仅使用 CPU 运行",
            "active_gpus_layer_split": "启用的 GPU (按层切分): {names}",
            "active_gpu": "启用的 GPU: {name} ({desc}) [{type}]",
        },

        # core/handlers.py
        "handlers": {
            # {missing} 清单项格式为 "显示名 (类名)"
            "handlers_unavailable": "以下 chat handler 在当前 llama-cpp-python 构建中不可用: {missing}",
            "thinking_unsupported": 'handler "{label}" 不支持思考, thinking 开关按关处理',
            "thinking_forced": 'handler "{label}" 为纯思考模型, thinking 开关按开处理',
        },

        # core/storage.py
        "storage": {
            "vram_cannot_fit_mmproj": "vram_limit ({vram_limit} GB) 装不下 mmproj 文件 (约 {mmproj_gb:.1f} GB), 主模型与 mmproj 均留在 CPU 以严格守预算",
            "vram_no_room_for_layer": "vram_limit ({vram_limit} GB) 装不下模型的任何一层 (每层约 {layer_size:.1f} GB), 主模型留在 CPU 以严格守预算",
            "kv_meta_fallback": "GGUF 注意力元数据不全, KV cache 改按文件体积折算, vram_limit 折算精度下降, 显存估算偏粗 (强量化模型的 KV 会被低估)",
            # 以下两条为 debug 级, 默认不输出
            "llm_close_failed": "llm 关闭失败: {e}",
            "handler_close_failed": "chat_handler 关闭失败: {e}",
            "free_vram_failed": "加载前释放 torch 显存失败: {e}",
            "preparing_mmproj": "正在准备 mmproj: {mmproj}",
            "loading_model": "正在加载模型: {model}",
            "load_params": "n_gpu_layers = {n_gpu_layers}, n_layer = {n_layer}, main_gpu = {main_gpu}, split_mode = {split_mode}",
            "load_failed_retry": "模型加载失败 ({e}), 释放 torch 显存后重试一次",
            "free_vram_retry_failed": "重试前释放 torch 显存失败: {free_err}",
            "cpu_only": "纯 CPU 推理: 模型层与 mmproj 均未上 GPU",
            "mmproj_only_gpu": "主模型全部层留在 CPU, 仅 mmproj (视觉编码器) 进显存 (落点由 mtmd 自选)",
            "cleanup_hook_applied": "模型清理钩子已挂载!",
        },

        # core/gguf_layers.py
        "gguf": {
            "parse_failed": "GGUF 解析失败: {e}",
            "block_count_missing": "GGUF 元数据中找不到 block_count",
        },

        # core/cqdm.py (tqdm 终端进度条的描述文字, 非 logger 输出)
        "cqdm": {
            "progress_desc": "处理中",
        },

        # shared/encoding.py
        "encoding": {
            "audio_batch_first_only": "收到含 {count} 段音频的 AUDIO 批次; 仅处理第一段",
        },

        # nodes/model/node_unload.py
        "unload": {
            "unloading": "正在卸载 llama 模型...",
        },

        # nodes/instruct/media/image/node_instruct.py
        "image_instruct": {
            "start_processing": "开始处理 {count} 张图片",
        },

        # nodes/bbox/ 各节点文件 + bbox_utils.py
        "bbox": {
            # {detail} 取下方两个 detail_* 变体之一
            "json_frame_mismatch": "JSON 结果 {json_count} 条与图像帧 {frame_count} 帧数量不符; 按索引配对, {detail}",
            "detail_extra_json": "多出的 JSON 条目复用最后一帧, 以单帧批次追加到 image_list",
            "detail_extra_frames": "未配对的末尾帧不画框, 原样透传",
            "draw_failed_json": "为 JSON #{i} 画框时出错: {e}",
            "segs_batch_first_frame": "BBoxes to SEGS 收到含 {batch_size} 张图像的批次; 裁剪图仅取自第一帧",
            # SEGS 与 MASK 两处同文
            "bbox_out_of_bounds": "跳过超出图像边界的 bbox: {bbox}",
            "bbox_empty_area": "跳过面积为空的 bbox: {bbox}",
            "no_cjk_font": "未找到 CJK 字体, bbox 标签可能渲染为方框",
            "bbox_draw_failed": "跳过绘制失败的 bbox ({label!r}: ({x0}, {y0}, {x1}, {y1})): {e}",
            "bbox_invalid_item": "跳过无效的 bbox 项: {bbox}",
            "bbox_non_numeric": "跳过坐标非数字的 bbox: {bbox}",
        },
    },
}
