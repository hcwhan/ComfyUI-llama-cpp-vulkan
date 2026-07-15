"""i18n 加载器: 按 LANGUAGE 常量加载对应语言文件, 导出 LANG 字典.

语言文件名含连字符 (language_zh-CN.py), 无法用常规 import 语句导入,
统一按文件路径加载; 所选语言文件缺失时打 warning 并回退默认英语
(LANGUAGE 笔误或部署不完整时不阻断插件加载), 默认语言文件也缺失时
立即抛错并列出可用语言 (英语文案缺失属部署损坏, 无可回退).
"""

import importlib.util
import logging
from pathlib import Path

# 切换 UI 语言只改这一行, 可选值即本目录下 language_*.py 的后缀: "en-US" / "zh-CN"
LANGUAGE = "zh-CN"

_DEFAULT_LANGUAGE = "en-US"
_I18N_DIR = Path(__file__).resolve().parent

# 加载器自身不能依赖语言文件, 回退警告文案例外地在此硬编码;
# LANGUAGE 本身无效 (笔误/不支持的值) 时查表同样落空, 用英文警告
_MISSING_FILE_WARNINGS = {
    "en-US": "[llama-cpp-vulkan] language file {name} not found, falling back to default language {default}, available languages: {available}",
    "zh-CN": "[llama-cpp-vulkan] 语言文件 {name} 不存在, 已回退到默认语言 {default}, 可用语言: {available}",
}


def _load_language(lang):
    path = _I18N_DIR / f"language_{lang}.py"
    if not path.is_file():
        available = ", ".join(sorted(p.stem.removeprefix("language_") for p in _I18N_DIR.glob("language_*.py")))
        if lang == _DEFAULT_LANGUAGE:
            raise FileNotFoundError(f"[llama-cpp-vulkan] default language file {path.name} not found, available languages: {available}")

        template = _MISSING_FILE_WARNINGS.get(lang, _MISSING_FILE_WARNINGS[_DEFAULT_LANGUAGE])
        # 与 shared/logger.py 取同名 logger, 不 import 项目内模块 (i18n 是最底层)
        logging.getLogger("llama-cpp-vulkan").warning(template.format(name=path.name, default=_DEFAULT_LANGUAGE, available=available))
        return _load_language(_DEFAULT_LANGUAGE)

    spec = importlib.util.spec_from_file_location(f"language_{lang.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LANG


LANG = _load_language(LANGUAGE)
