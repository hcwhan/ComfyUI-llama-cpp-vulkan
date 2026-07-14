"""LLM 输出文本处理: 代码围栏剥离, 逐张结果拆分, JSON 解析, 嵌套取值. 被 util 文本节点与 bbox 节点共同使用."""

import json
import re

# 开头的 ```label(标签限单词类字符,可无,如 json/python/c++;CommonMark
# 允许标签前有空格,少数模型会输出 "``` json" 形态);结尾的 ```。
# 两端独立匹配,生成被截断导致围栏未闭合时,开头的标记仍能剥离。
# 标签不能用 [^\s`]* 之类的宽匹配:围栏后无换行直接跟正文时会把正文吞掉
_FENCE_OPEN_RE = re.compile(r"^```[ \t]*[\w+.-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```$")
# 文本中部的完整围栏块(用于 "前导说明 + 围栏块" 形态的回退提取)
_FENCE_BLOCK_RE = re.compile(r"```[ \t]*[\w+.-]*[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL)


def strip_code_fence(text):
    """去除 LLM 输出首尾的 ```label ... ``` 代码块标记.

    兼容任意标签和裸 ``` 围栏:
    模型即使被要求输出 json 也可能给出不带标签的围栏.
    模型输出 "好的, 结果如下:" 之类前导说明时首部不是围栏,
    此时回退为提取文本中第一个完整围栏块的内容.
    """
    text = text.strip()
    stripped = _FENCE_OPEN_RE.sub("", text)
    if stripped != text:
        return _FENCE_CLOSE_RE.sub("", stripped)
    block = _FENCE_BLOCK_RE.search(text)
    if block:
        return block.group(1)
    return _FENCE_CLOSE_RE.sub("", text)


# image Instruct 逐张模式在多图结果间插入的分隔行(独占一行, 行首行尾锚定,
# 降低正文文本误匹配的概率; JSON 文本中换行均为 \n 转义, 不会产生真实分隔行)
_IMAGE_SEP_RE = re.compile(r"^====== Image \d+ ======[ \t]*\r?$", re.MULTILINE)


def split_image_results(text):
    """把 image Instruct 逐张模式的拼接输出按分隔行拆回逐张结果列表.

    无分隔行时返回单元素列表(整段原文), 普通文本 / 单图结果可安全通过.
    只丢弃首个分隔行之前的空首段; 中间/末尾的空结果保留为空字符串占位,
    维持 "第 i 段对应第 i 张图" 的索引对齐(下游按索引配对画框/拆列表).
    """
    if not _IMAGE_SEP_RE.search(text):
        return [text]
    parts = [p.strip() for p in _IMAGE_SEP_RE.split(text)]
    if not parts[0]:
        parts = parts[1:]
    return parts


def parse_json(json_str):
    try:
        parsed = json.loads(strip_code_fence(json_str))
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed


def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split(".")
    for key in keys:
        if isinstance(data, str):
            # 嵌套的 JSON-in-string 字段:解析失败视为无法下钻,回落 default,
            # 与 "key 不存在" 的语义一致(顶层输入的解析错误由 parse_json 报出)
            try:
                data = json.loads(data)
            except ValueError:
                return default
        if isinstance(data, dict) and key in data:
            data = data[key]
        elif isinstance(data, list):
            # 数组按数字下标下钻(如 items.0.label), 支持负下标从尾部取;
            # 非数字 key 或越界与 "key 不存在" 同语义, 回落 default
            try:
                index = int(key)
            except ValueError:
                return default
            if not -len(data) <= index < len(data):
                return default
            data = data[index]
        else:
            return default
    return data
