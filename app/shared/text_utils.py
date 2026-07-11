"""LLM 输出文本处理: 代码围栏剥离, JSON 解析, 嵌套取值. 被 util 文本节点与 bbox 节点共同使用."""

import re
import json

# 开头的 ```label(标签限单词类字符,可无,如 json/python/c++);结尾的 ```。
# 两端独立匹配,生成被截断导致围栏未闭合时,开头的标记仍能剥离。
# 标签不能用 [^\s`]* 之类的宽匹配:围栏后无换行直接跟正文时会把正文吞掉
_FENCE_OPEN_RE = re.compile(r"^```[\w+.-]*[ \t]*\r?\n?")
_FENCE_CLOSE_RE = re.compile(r"\r?\n?```$")


def strip_code_fence(text, label=""):
    """去除 LLM 输出首尾的 ```label ... ``` 代码块标记。

    label 仅为语义提示,实际兼容任意标签和裸 ``` 围栏:
    模型即使被要求输出 json 也可能给出不带标签的围栏。
    """
    text = text.strip()
    text = _FENCE_OPEN_RE.sub("", text)
    return _FENCE_CLOSE_RE.sub("", text)


def parse_json(json_str):
    try:
        parsed = json.loads(strip_code_fence(json_str, "json"))
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed


def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
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
        else:
            return default
    return data
