"""i18n 加载器: 按 LANGUAGE 常量加载对应语言文件, 导出 LANG 字典.

语言文件名含连字符 (language_zh-CN.py), 无法用常规 import 语句导入,
统一按文件路径加载; 文件缺失时立即抛错并列出可用语言, 不静默回退
(文案缺失属部署错误, 回退只会把问题拖到更难定位的节点执行期).
"""

import importlib.util
from pathlib import Path

# 切换 UI 语言只改这一行, 可选值即本目录下 language_*.py 的后缀: "zh-CN" / "en-US"
LANGUAGE = "zh-CN"

_I18N_DIR = Path(__file__).resolve().parent


def _load_language(lang):
    path = _I18N_DIR / f"language_{lang}.py"
    if not path.is_file():
        available = ", ".join(sorted(p.stem.removeprefix("language_") for p in _I18N_DIR.glob("language_*.py")))
        raise FileNotFoundError(f"[llama-cpp-vulkan] 语言文件 {path.name} 不存在, 可用语言: {available}")
    spec = importlib.util.spec_from_file_location(f"language_{lang.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LANG


LANG = _load_language(LANGUAGE)
