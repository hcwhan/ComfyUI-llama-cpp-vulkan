"""任务预设 prompt 模板池, 全模态共享, 显示范围由每个预设的 use 字段声明.

每个预设是 {"use": [模态...], "content": 模板文本}:
- use      声明该预设在哪些 Instruct 节点显示(text/image/video/audio),
           节点类通过 MODALITY 类属性对号入座
- content  模板文本; dict 声明顺序即各模态 UI 下拉框的顺序(按 use 过滤后),
           且各模态过滤结果的第一项就是该节点的默认预设(不另行声明)

模块加载时校验 use 中的模态名, 拼写错误在启动期即报错.

模板占位符(三字符形式, 避免与正文中的 @/# 字符冲突):
- "@@@"  按节点模态替换(图像/视频/音频; text 节点按文生图语境替换为 图像);
         模板中占位符两侧留空格以提升可读性, 空格随替换保留, 不影响模型理解
- "###"  由 custom_prompt 填充(模板含 "###" 时 custom_prompt 必填, 如 BBox 检测的
         目标类别、待改写的提示词); 此类预设按约定在名字后标注 "(需custom_prompt)",
         该标注仅为 UI 提示, 判定依据是模板内容而非名字

预设与 custom_prompt 组合的生效场景(instruct.py 的 _build_user_prompt):
1. 预设为空白("空白 - 空") + custom_prompt 非空 -> 纯自定义, 只发送 custom_prompt
2. 预设为空白("空白 - 空") + custom_prompt 为空 -> user 文本为空, 只发送媒体内容
   (适合 chat 模板自带默认指令的模型; text 节点无媒体载荷, 此场景在
   _run 中按 REQUIRE_USER_TEXT 拦截报错, 不白白加载模型)
3. 模板不含 "###" + custom_prompt 为空   -> 使用预设模板原文("@@@" 替换为模态词)
4. 模板不含 "###" + custom_prompt 非空   -> custom_prompt 整体覆盖预设, 模板被丢弃
5. 模板含 "###"   + custom_prompt 非空   -> custom_prompt 填入 "###" 占位符
6. 模板含 "###"   + custom_prompt 为空   -> 报错, 提示必须填写 custom_prompt
(场景 1/2 是 4/3 在空模板下的特例, 代码中走同一分支)
"""

user_prompt_presets = {
    "空白 - 空": {
        "use": ["text", "image", "video", "audio"],
        "content": '',
    },
    "常规 - 描述": {
        "use": ["image", "video", "audio"],
        "content": '描述这个 @@@ 。',
    },
    "常规 - 转写": {
        "use": ["audio"],
        "content": '转写这段 @@@ 的内容。',
    },
    "创意 - 提示词增强 (需custom_prompt)": {
        "use": ["text"],
        "content": '改写并增强下面的用户提示词，用于文生 @@@ 创作。保留原意与关键词，使其更具表现力、视觉细节更丰富。**只输出改写后的提示词文本**，不要输出推理过程或任何额外说明。\n\n用户提示词："###"',
    },
    "创意 - 图片分析": {
        "use": ["image"],
        "content": '详细描述这个 @@@ ，将主体、服饰、配饰、背景与构图分节说明。',
    },
    "创意 - 视频总结": {
        "use": ["video"],
        "content": '总结这段视频中的关键事件与叙事要点。',
    },
    "创意 - 短篇故事": {
        "use": ["image", "video"],
        "content": '以这个 @@@ 为灵感，写一篇富有想象力的短篇故事。',
    },
    "提示词风格 - 标签": {
        "use": ["image", "video"],
        "content": '你的任务是仅基于 @@@ 中的视觉信息，为文生 @@@ AI 生成一份以逗号分隔的标签列表。最多输出 50 个不重复的标签。只描述主体、服饰、环境、色彩、光线、构图等视觉元素，不要包含抽象概念、主观解读、营销词汇或技术术语（例如不要出现 "SEO"、"品牌契合"、"病毒传播潜力"）。目标是一份简洁的视觉描述标签列表，避免重复标签。',
    },
    "提示词风格 - 简洁": {
        "use": ["image", "video"],
        "content": '分析这个 @@@ ，生成一条简洁的单句文生 @@@ 提示词，简明描述主体与场景。',
    },
    "提示词风格 - 详细": {
        "use": ["image", "video"],
        "content": '基于这个 @@@ 生成一条详细而有艺术感的文生 @@@ 提示词。把主体、动作、环境、光线与整体氛围融合为一段 2-3 句的连贯文字，突出关键视觉细节。',
    },
    "提示词风格 - 极致详细": {
        "use": ["image", "video"],
        "content": '基于这个 @@@ 生成一条极其详细的文生 @@@ 提示词。用一段丰富的文字，细致刻画主体外貌、服饰质感、背景元素、光线的质感与色彩、阴影以及整体氛围，力求高度描述性与沉浸感。',
    },
    "提示词风格 - 电影感": {
        "use": ["image", "video"],
        "content": '作为资深提示词工程师，为 @@@ 生成 AI 创作一条高度详细、富有感染力的提示词。描述主体、姿态、环境、光线、氛围与艺术风格（如写实摄影、电影感、绘画感），并将所有元素编织为一段自然流畅的文字，注重视觉冲击力。',
    },
    "视觉 - BBox 目标检测 (需custom_prompt)": {
        "use": ["image"],
        "content": '定位属于以下类别的所有目标："###"。以 JSON 列表输出边界框坐标，格式为 {"bbox_2d": [x1, y1, x2, y2], "label": "string"}。',
    },
}

_MODALITIES = {"text", "image", "video", "audio"}

for _name, _spec in user_prompt_presets.items():
    _unknown = set(_spec["use"]) - _MODALITIES
    if _unknown:
        raise ValueError(f'Preset "{_name}" uses unknown modality: {", ".join(sorted(_unknown))}')


def instruct_presets(modality):
    """返回某模态可用的预设名列表, dict 声明顺序即 UI 下拉框顺序."""
    return [name for name, spec in user_prompt_presets.items() if modality in spec["use"]]


def preset_content(name):
    """按预设名取模板文本, 未知名字抛带指引的 ValueError.

    预设改名/删除后, 经连线传入的旧工作流值可能失配 (widget 常量会先被
    ComfyUI 的 combo 前置校验拦截), 报错与 resolve_config 的 unknown
    handler 风格对齐, 不暴露裸 KeyError.
    """
    try:
        return user_prompt_presets[name]["content"]
    except KeyError:
        raise ValueError(f'Unknown preset_prompt: "{name}". Re-select a preset from the dropdown (the workflow may reference a renamed or removed preset).') from None
