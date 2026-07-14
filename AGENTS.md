# ComfyUI-llama-cpp-vulkan

基于 llama.cpp 的 ComfyUI LLM/VLM 自定义节点插件, 使用 Vulkan 实现跨平台 GPU 加速推理.

## 项目概况

- **核心依赖**: llama-cpp-python (自编译 Vulkan ABI3 wheel, v0.3.42)
- **GPU 后端**: Vulkan (非 CUDA/ROCm, 独立于 PyTorch 的 GPU 推理路径)
- **支持平台**: Windows / Linux(预编译 Vulkan wheel 仅覆盖这两个平台; Linux 为 manylinux_2_31, 要求 glibc >= 2.31)
- **ComfyUI 节点类别**: `llama-cpp-vulkan`

## 目录结构

组织原则: `src/nodes/` 只放节点声明(文件名一律 `node_` 前缀), 与某个节点强相关的工具放在该节点文件旁边(如 `bbox_utils.py`, `system_prompt_presets.py`, media 共用的 `encoding.py`); 跨领域复用的逻辑放 `src/core/`(模型生命周期, 推理骨架, 基础设施)与 `src/shared/`(纯工具).

```
ComfyUI-llama-cpp-vulkan/
  __init__.py                 # 入口: from .src.nodes 导出 NODE_CLASS_MAPPINGS
  pyproject.toml              # 项目元数据, 依赖声明
  requirements.txt            # pip 依赖(含平台条件 llama-cpp-python wheel URL)
  复核结论.md                 # 复核结论存档(见"文档维护原则")
  .github/workflows/
    build-vulkan-wheels-abi3.yml  # CI: 构建/发布双平台 Vulkan ABI3 wheel
  docs/
    项目分析.html             # 历史快照(页头已注明生成 commit, 仅供历史参考)
  src/
    core/                     # 核心逻辑(非节点)
      storage.py              #   模型生命周期: 全局单例, resolve_config 校验, 显存折算, unload 钩子
      instruct.py             #   Instruct 基类(text 骨架)+ media 基类(VLM 校验)+ 中断/thinking/hybrid 工具
      prompts.py              #   任务预设模板池(@@@/### 占位符, use 字段声明适用模态)
      devices.py              #   Vulkan GPU 设备检测与选择(ggml C API / ctypes)
      handlers.py             #   Chat handler 注册表(数十种 VLM 格式)
      model_paths.py          #   llm/LLM 模型目录注册与路径查找
      gguf_layers.py          #   GGUF 文件解析: 读取模型层数 (block_count)
      cqdm.py                 #   进度条封装: 同时驱动 ComfyUI ProgressBar 和 tqdm
    shared/                   # 跨领域纯工具(非节点)
      logger.py               #   插件统一 logging logger(沿用 ComfyUI 的日志配置)
      text_utils.py           #   代码围栏剥离, 逐张结果拆分, JSON 解析, 嵌套取值
      types.py                #   AnyType 万能透传类型
    nodes/                    # 只放节点声明
      __init__.py             #   节点注册表(16 个节点的映射)
      model/
        node_loaders.py       #   llm / vlm 两个 Model Loader
        node_parameters.py    #   llama.cpp Parameters
        node_unload.py        #   llama.cpp Unload Model
      type/
        text/
          node_instruct.py    #   llama.cpp text Instruct
        media/
          encoding.py         #   media 共用: 张量/音频转 base64, 缩放
          image/
            node_instruct.py  #   llama.cpp image Instruct
          video/
            node_instruct.py  #   llama.cpp video Instruct
          audio/
            node_instruct.py  #   llama.cpp audio Instruct
          bbox/
            node_bbox.py      #   BBox 工具链 4 节点
            bbox_utils.py     #   BBox 强相关工具: 坐标换算, 画框, 羽化 mask
      util/
        node_parse_json.py    #   Parse JSON
        node_remove_code_block.py  # Unpack Code Block
        node_split_output.py       # Split Instruct Output
        node_system_prompt.py      # System Prompt Preset
        system_prompt_presets.py   # 中文 Prompt 增强系统提示词模板池(Qwen-Image/Z-Image/Flux.2/Wan)
  tools/
    check_devices.py          # 独立诊断脚本: 列出 GGML 后端检测到的所有设备
  tests/                      # 单元测试(标准库 unittest, 用 ComfyUI 嵌入式 Python 运行)
    comfy_stubs.py            #   comfy/folder_paths 最小替身, 满足 import 期依赖
```

## 数据流与类型隔离

节点全集见 `src/nodes/__init__.py` 注册表.

```
llama_cpp_llm_model_loader --> LLAMACPPLLM ------> llama_cpp_text_instruct
llama_cpp_vlm_model_loader --> LLAMACPPVLM --+---> llama_cpp_image_instruct
                                             +---> llama_cpp_video_instruct
llama_cpp_parameters ----> LLAMACPPARAMS ----+---> llama_cpp_audio_instruct
                        (全部 Instruct 的可选输入)
system_prompt_preset ----> STRING -----------+   (接全部 Instruct 的 system_prompt)

全部 Instruct 输出单端口 STRING (output);
image 逐张模式的多图结果以 "====== Image N ======" 分隔行拼接
    |
    +--> parse_json_node / remove_code_block
    +--> split_instruct_output    (拆回 STRING 列表, 接第三方列表语义节点)
    +--> json_to_bboxes           (内建同款拆分, output 可直连)
              |
              +--> BBOX --+--> bboxes_to_bbox  (按索引选取, 输出仍为 BBOX)
                          +--> bboxes_to_segs / bboxes_to_mask  (下游图像处理)
```

`llama_cpp_unload_model` 为 any 透传节点, 可串接在任意连线上, 不参与上图数据流.

`LLAMACPPLLM`(llm Loader 输出)与 `LLAMACPPVLM`(vlm Loader 输出)完全独立: llm 配置只能连 text Instruct, vlm 配置只能连 image/video/audio Instruct, 连错在连线阶段即被 ComfyUI 类型系统拦截. 两种配置 dict 结构相同(llm 侧 mmproj/chat_handler 固定为 "None"), 底层共用 `core/storage.py` 的加载路径.

## 架构要点

### GPU 设备管理

`src/core/devices.py` 通过 ggml C API (ctypes) 直接枚举 Vulkan GPU 设备, 区分独显 (GPU) 和核显 (IGPU). 这是独立于 PyTorch/CUDA 的 Vulkan 推理路径.

关键函数链: `_detect_gpu_devices()` -> `_selectable_devices()` -> `gpu_device_choices` / `resolve_device_selection()`

选择语义(与 llama.cpp 构建 `model->devices` 的规则对齐):

- llama.cpp 只把独显按枚举顺序收入设备列表, 仅当无独显时才收第一个核显, 其余设备无法通过 `main_gpu` 到达, 因此下拉框只展示可达设备
- `Auto (独显优先)`: 走 llama.cpp 默认行为(`LLAMA_SPLIT_MODE_LAYER`, 独显优先, 多独显按层切分)
- 显式选择某设备: 传 `LLAMA_SPLIT_MODE_NONE` + 该设备在可选列表中的索引, 整个模型加载到单卡; `main_gpu` 在 LAYER 模式下会被 llama.cpp 忽略, 这是显式选择必须切 NONE 的原因

### 模型生命周期

`src/core/storage.py` 的 `LLAMA_CPP_STORAGE` 类管理全局单例模型状态:

- 懒加载: 两个 Loader 只调用 `resolve_config()` 做快速失败校验(模型/mmproj 路径存在, mmproj 与 chat_handler 配对合法)并返回 config, 实际加载由 Instruct 节点按需触发; 多组 loader+instruct 交错时避免全局单例被 loader 反复挤占
- `load_model()`: 先 `resolve_config()` 校验再卸载旧模型(无效配置不影响已加载的模型), 随后加载 GGUF 模型; 可选的 mmproj(视觉编码器)在此处只构造 chat_handler(校验路径), 真正加载进显存由 mtmd 在首次推理时惰性初始化(与加载前的 `mm.free_memory` 腾挪同在一次节点执行内, 时序有效)
- `clean()`: 释放模型和 chat_handler 资源
- 通过 monkey-patch `mm.unload_all_models` 实现 ComfyUI 模型卸载(前端 Free 按钮 / OOM 处理)时自动清理
- `vram_limit` 折算 `n_gpu_layers` 集中在 `_estimate_n_gpu_layers()`: 每层显存 =(文件体积 x (1 + 固定开销系数) + KV cache 字节数)/ 层数. KV 按 `core/gguf_layers.py` 解析的注意力元数据(`head_count_kv`/`embedding_length` 等, hybrid 模型的逐层数组取均值)精确计算, 与权重量化无关; 元数据不全时回退 `_vram_factor(n_ctx)` 体积折算经验系数(n_ctx=8192 时合计 1.55, 对强量化模型会低估 KV). mmproj 体积先从预算中扣除, 预算装不下 mmproj 时两者全留 CPU; 扣除后不足主模型 1 层时主模型留 CPU, mmproj 照常进显存(全部分支严格守预算, 不足 1 层不强制上卡)

### Instruct 继承体系

`src/core/instruct.py` 提供两级基类, 四个 Instruct 节点只声明 `INPUT_TYPES` 与模态专属的 runner 闭包:

- `llama_cpp_instruct_base`: 通用骨架. `_run()` 负责组消息(system + user), 复制采样参数, `InterruptWatcher` 监视, force_offload / hybrid KV 重置收尾; `seed_input()/prompt_inputs()/runtime_inputs()/optional_inputs()` 是 INPUT_TYPES 字段组装块
- `llama_cpp_media_instruct_base`: 多模态骨架, `MODEL_TYPE = "LLAMACPPVLM"`, 附 `require_mmproj()` 兜底校验

### 任务预设系统

预设模板池在 `core/prompts.py`, 任务预设与增强预设文本均为中文:

- 显示范围由每个预设的 `use` 字段声明(text/image/video/audio), 节点类通过 `MODALITY` 类属性对号入座; dict 声明顺序即 UI 下拉框顺序, 各模态过滤结果的第一项即该节点的默认预设
- `@@@` 占位符替换为节点类的 `MEDIA_WORD`(text/image -> 图像, video -> 视频, audio -> 音频)
- `###` 占位符由 custom_prompt 填充: 模板含 `###` 时必填, 否则非空 custom_prompt 整体覆盖预设

### Chat Handler 注册表

`src/core/handlers.py` 的 `_HANDLER_SPECS` 表集中定义全部 handler, 数据形态, 排序约定与新增须知见该表头注释. 要点: 构造期固定参数(thinking 开关, Generic-MTMD 兜底的 `chat_format` 等)经 `functools.partial` 预绑定进 `HANDLERS` 的构造器, `storage.py` 因此不感知 thinking 逻辑; 启动时解析失败的类只从下拉框剔除并打 warning(防御 wheel 升级时的类变动, 不静默, 不阻断 import); `-Thinking` 后缀与开关值的一致性由 `tests/test_handlers.py` 契约测试锁定.

### 多模态输入

- image Instruct: `batch_images` 开关切换逐张推理(多图结果以 `====== Image N ======` 分隔行拼接, `split_instruct_output` 节点与 `json_to_bboxes` 的内建拆分均可还原为逐张列表)与批量单请求; 批量多图时缩放到 `max_size`, 单图保持原分辨率
- video Instruct: `frames` 输入为 IMAGE 帧批次(ComfyUI 生态的视频通行形态), 按 `max_frames` linspace 均匀抽帧后缩放, 并在 system prompt 前注入"连续视频"语义提示
- audio Instruct: ComfyUI `AUDIO` dict 由 `media/encoding.py` 的 `audio2base64()` 均值混为单声道 16-bit WAV, 以 `input_audio` 内容项注入(重采样由 llama.cpp 的 mtmd 解码端完成), 服务 Qwen3-ASR 等音频 handler; 音频是否被 mmproj 支持由 llama-cpp-python 侧校验

### 推理输出与中断

- 无会话状态: 每次执行都是全新的一次性请求(system prompt + 本次提问), 不保留任何跨执行的对话历史
- `strip_thinking` 开关(默认开): 剥离三种思考形态 - `<think>...</think>` 推理块(兼容 generation prompt 已注入 `<think>` 导致输出只含闭合标签的情况), Gemma4 的 channel 格式(取最后一个 `<channel|>` 之后, 覆盖 E 系列无开标签的纯文本思考形态), GLM-4.1V 的 `<answer>` 包裹(handler 以 `</answer>` 为 stop token 导致开标签残留); 未闭合(生成截断)时均保持原样
- `InterruptWatcher`: 推理期间守护线程每 200ms 轮询 `mm.processing_interrupted()`, 命中后调用 `Llama.abort()` 使生成立即停止; llama-cpp-python 在每次请求开始会 clear abort 事件, 因此监视线程命中后持续重复 set 以抗竞态
- 每次节点执行结束后按 `is_hybrid_arch()`(`_model.is_hybrid()`/`is_recurrent()` C API)判断是否整体重置 KV cache(重置在 `_run()` 的 finally 中, image 逐张模式中间的多次请求之间不重置, 依赖 wheel 内置的 hybrid checkpoint 前缀匹配): hybrid/recurrent 架构(Qwen3.5, LFM2 系等)的线性注意力状态无法跨请求前缀复用; 纯 SWA 模型(Gemma3)不受影响

## 修改代码须知

### 文档维护原则

- 本文件只记录无法从代码直接看出的内容: 架构决策, 跨文件约束, 易踩的坑. 一般性内容(如"在某个 dict 加一行"式的操作步骤, 读代码即可自然得出的说明)不要写入本文件
- 根目录 `复核结论.md` 是复核结论存档, 用于避免重复排查或重新争论. 只收录两类条目: 项目外事实与跨边界对接结论(经明确源码分析确立, 对照 wheel / ComfyUI / llama.cpp 上游 / 下游生态实际源码核实), 设计决策/权衡存档; 项目内代码行为不在其中复述, 以代码与注释本身为准. 写入前需与用户确认

### Commit message 规范

- 一律使用中文书写(标题与正文), 保留 conventional commits 类型前缀(`feat:` / `fix:` / `refactor:` / `docs:` / `chore:` / `ci:` / `test:`, 破坏性变更加 `!`)
- 代码标识符, 文件名, API 名等专有名词保持原文, 不翻译
- 提交用消息文件而非 `-m` 内联: Windows PowerShell 下 `-m` 中的引号, heredoc 与以 `-` 开头的词会被参数解析破坏. 先把消息写入临时文件, 再 `git commit -F <文件>`, 提交后删除临时文件

### 标点符号规范

- 全项目一律使用英文半角标点, 覆盖代码注释, docstring, 节点文案 (tooltip/placeholder/报错消息), 预设模板文本与 Markdown 文档
- `,` `.` `:` (以及 `;` `!` `?`) 后接一个空格; 位于行尾, 或后跟闭合符 (引号, 括号) 时不加
- 引号一律使用英文引号: 常规引用用双引号 "; 嵌套场景 (如预设模板中示例包裹层与图中渲染文字) 相邻层级以单引号 ' 区分
- 破折号 (中式双破折号 "——" 与 em dash "—") 一律写作两侧带空格的中划线 " - "

### Python 文件规范

- 每个 .py 文件顶部必须有描述整个文件用途的模块 docstring, docstring 之后空一行再写代码
- 无实际代码的包不创建 `__init__.py`(子包走 Python 隐式命名空间包); 仅根入口, `src/nodes/__init__.py` 注册表与 `tests/__init__.py`(测试导入引导)三处有 `__init__.py`

### 依赖版本对接原则

项目代码只对接 `requirements.txt` 中固定的依赖版本(特别是 llama-cpp-python 的 JamePeng Vulkan wheel), 不为历史版本或官方构建编写兼容/回退代码. mmproj 路径统一用 `mmproj_path` 键传入 handler.

当前依赖的 wheel 对接面清单(升级 wheel 时按单复核, 与 `tests/test_wheel_contract.py` 一一对应): `llm.n_tokens`, `llm._ctx.memory_clear`, `llm._hybrid_cache_mgr`, `llm._model.is_hybrid()/is_recurrent()`, `llama_cpp._ggml`(设备枚举符号), chat handler 的 `mmproj_path` 实例属性(`require_mmproj` 的判定依据, getattr 兜底使重命名不报错而是恒判"未配置"), `Llama.close()/abort()`(公开方法, 但同为按单复核的对接面), `llama_cpp.llama_cpp.llama_split_mode` 枚举(NONE/LAYER), `create_chat_completion` 接受 Parameters 节点全部 12 个采样参数与 `seed`. 契约测试静态检查上述接口存在性(不加载模型), 升级 wheel 后运行即可发现断裂; 公开 handler 类的契约另由 `tests/test_handlers.py` 锁定.

### INPUT_TYPES 字段顺序

修改代码时不考虑兼容旧工作流(ComfyUI 的 widget 值按 `INPUT_TYPES` 声明顺序序列化, 调整顺序会使已保存工作流的 widget 值错位, 属预期代价), 只考虑代码本身的合理性. 特别是后加的配置项, 不要为迁就旧工作流而堆在末尾, 应按语义放到合理位置(与相关字段分组).

Instruct 子类的字段顺序约定: 模型端口 -> 媒体输入 -> `seed_input()` -> `prompt_inputs()` -> 模态专属字段 -> `runtime_inputs()`(strip_thinking -> force_offload, 收尾动作垫底).

### Wheel 构建与发布 (CI)

`.github/workflows/build-vulkan-wheels-abi3.yml` 手动触发(workflow_dispatch, 输入 JamePeng/llama-cpp-python 的 ref), 并行构建 Windows + Linux 两个 ABI3 wheel 并自动发布 GitHub Release. 要点:

- `+vulkan` 本地版本在构建前注入 `llama_cpp/__init__.py` 的 `__version__`, wheel 文件名与 METADATA 天然一致(不做构建后重命名)
- 两平台均开启 `GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS`(CPU 后端按指令集运行时分发)
- Windows: Vulkan SDK 走 action 缓存(stripdown 后缓存), 编译走 sccache; 冷缓存约 25 分钟, 热缓存约 5 分钟
- Linux: Vulkan 头文件/glslc/loader 来自 conda-forge(不下载 LunarG SDK tarball); `CMAKE_PREFIX_PATH=/opt/vulkan` 是 `find_package(SPIRV-Headers)` 的必需项, 不能删
- Linux repair 目标为 `manylinux_2_31`(gcc-toolset-14 产物引用 GLIBCXX_3.4.25, 超出 2_28 白名单), 且 repair 的 `LD_LIBRARY_PATH` 不能包含 `/opt/vulkan/lib`(避免 auditwheel 解析到 conda 的新版 libstdc++)
- 发布新 wheel 后需同步更新 `requirements.txt` 的两个 URL 和两个 README 的平台说明

## 已知问题

1. **单模型实例**: `LLAMA_CPP_STORAGE` 是全局单例, 不支持同时加载多个模型
2. **核显不可选(有独显时)**: llama.cpp 的设备收集规则决定了有独显时核显无法通过 `main_gpu` 选中; 如需强制核显推理, 只能在进程启动前设置 `GGML_VK_VISIBLE_DEVICES` 环境变量(devices.py import 时即初始化 Vulkan, 之后设置无效)
3. **import 时初始化 Vulkan**: 设备枚举在插件加载时同步执行(约几百 ms), 属有意设计(UI 下拉框需要启动期确定设备列表)
4. **mmproj 不跟随显式选卡**: mtmd 的 `mtmd_context_params` 只有 `use_gpu` 布尔开关, 无设备索引字段, 多卡下显式选择非默认 GPU 时视觉编码器仍落在 mtmd 默认挑选的设备上(上游限制); 插件在 `vram_limit=0`(纯 CPU)或 mmproj 折算体积超出 vram_limit 预算时传 `use_gpu=False` 让 mmproj 一并留在 CPU(严格守预算)
5. **多分片 GGUF (split shards) 不支持**: 显存折算用 `os.path.getsize` 只计所选文件体积, 层数却是全模型 `block_count`, 选首分片加载(llama.cpp 会自动带上其余分片)时 `n_gpu_layers` 会被高估; 模型下拉框也会列出 `-00002-of-00003.gguf` 等非首分片, 选中会加载失败. 按不支持处理, 如需使用请先以 `llama-gguf-split --merge` 合并为单文件
