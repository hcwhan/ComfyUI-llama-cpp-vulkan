"""i18n 加载器: 解析目标语言并加载对应语言文件, 导出 LANG 字典.

语言解析 (LANGUAGE 为 "auto" 时) 按三级优先, 设置读写统一走 locale_settings.py:
1. 实时读 ComfyUI 语言设置 Comfy.Locale (读取与落盘见 read_comfy_locale);
2. Comfy.Locale 缺失 (用户从未设置过语言, 前端按浏览器语言自动检测,
   服务端无从得知) 时, 读插件根 settings.json 的 language.frontend_locale -
   上次会话前端上报的实际显示语言 (上报链路见 core/locale_sync.py);
3. 都没有时回退默认英语. 短码经 _LOCALE_TO_LANGUAGE 映射到语言文件,
   未映射的短码 (无对应文案) 同样落默认英语.
LANGUAGE 为具体语言代码时强制使用, 跳过全部读写. 节点文案在插件 import 期
固化, 多用户模式下无法按用户区分语言, 只读 default 用户的设置 (尽力而为).

语言文件名含连字符 (language_zh-CN.py), 无法用常规 import 语句导入,
统一按文件路径加载; 所选语言文件缺失时打 warning 并回退默认英语
(LANGUAGE 笔误或部署不完整时不阻断插件加载), 默认语言文件也缺失时
立即抛错并列出可用语言 (英语文案缺失属部署损坏, 无可回退).
"""

import importlib.util
import logging
from pathlib import Path

from . import locale_settings

# "auto" 跟随 ComfyUI 设置界面的语言选项 (Comfy.Locale); 需强制指定语言时
# 改为本目录下 language_*.py 的后缀: "en-US" / "zh-CN"
LANGUAGE = "auto"

_DEFAULT_LANGUAGE = "en-US"
_I18N_DIR = Path(__file__).resolve().parent

# ComfyUI 前端 Comfy.Locale 短码 -> 本目录语言文件后缀. zh-TW 归入 zh-CN
# (繁体用户读简体优于英语); 其余前端短码 (ja/ko/fr/ru/es) 无对应文案,
# 与检测失败一并落默认英语
_LOCALE_TO_LANGUAGE = {
    "en": "en-US",
    "zh": "zh-CN",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-CN",
}

# 加载器自身不能依赖语言文件, 回退警告文案例外地在此硬编码;
# LANGUAGE 本身无效 (笔误/不支持的值) 时查表同样落空, 用英文警告
_MISSING_FILE_WARNINGS = {
    "en-US": "[llama-cpp-vulkan] language file {name} not found, falling back to default language {default}, available languages: {available}",
    "zh-CN": "[llama-cpp-vulkan] 语言文件 {name} 不存在, 已回退到默认语言 {default}, 可用语言: {available}",
}


def _resolve_language():
    """返回目标语言代码: LANGUAGE 非 "auto" 时原样返回, 否则按三级优先解析 (见模块 docstring)."""
    if LANGUAGE != "auto":
        return LANGUAGE

    locale = locale_settings.read_comfy_locale()
    if locale is not None:
        return _LOCALE_TO_LANGUAGE.get(locale, _DEFAULT_LANGUAGE)

    frontend_locale = locale_settings.get_language_setting("frontend_locale")
    if frontend_locale is not None:
        return _LOCALE_TO_LANGUAGE.get(frontend_locale, _DEFAULT_LANGUAGE)

    return _DEFAULT_LANGUAGE


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


LANG = _load_language(_resolve_language())
