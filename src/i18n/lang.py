"""i18n 加载器: 解析目标语言并加载对应语言文件, 导出 LANG 字典.

语言解析 (LANGUAGE 为 "auto" 时) 按三级优先, 设置读写统一走 locale_settings.py:
1. 实时读 ComfyUI 语言设置 Comfy.Locale (读取与落盘见 read_comfy_locale);
2. Comfy.Locale 缺失 (用户从未设置过语言, 前端按浏览器语言自动检测,
   服务端无从得知) 时, 读插件根 settings.json 的 language.frontend_locale -
   上次会话前端上报的实际显示语言 (上报链路见 core/locale_sync.py);
3. 都没有时回退默认英语. 短码经 _LOCALE_TO_LANGUAGE 映射到语言文件,
   未映射的短码 (无对应文案) 同样落默认英语; 非字符串短码 (设置文件被
   外力写坏) 视同缺失, 继续下一级 (list/dict 不可哈希, 若直接查表会抛
   TypeError 阻断插件加载).
LANGUAGE 为具体语言代码时强制使用, 跳过全部读写. 节点文案在插件 import 期
固化, 多用户模式下无法按用户区分语言, 只读 default 用户的设置 (尽力而为).
加载完成后以解析结果语言打一条 info 日志 (语言代码与命中的解析来源),
供排查界面语言不符时定位到具体层级.

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
# (繁体用户读简体优于英语); 其余前端短码 (ja/ko/fr 等) 无对应文案,
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

# 启动期解析结果日志 (硬编码理由同上); {source} 为命中的解析层级,
# 取 _resolve_language 返回的功能性标识 (LANGUAGE / Comfy.Locale /
# frontend_locale / default), 不翻译
_RESOLVED_INFO = {
    "en-US": "[llama-cpp-vulkan] UI language: {lang} (source: {source})",
    "zh-CN": "[llama-cpp-vulkan] 界面语言: {lang} (来源: {source})",
}


def _resolve_language():
    """返回 (目标语言代码, 解析来源标识): LANGUAGE 非 "auto" 时原样返回, 否则按三级优先解析 (见模块 docstring).

    来源标识指命中的解析层级 (供启动日志显示); 短码未映射到语言文件时
    语言代码落默认英语, 来源仍如实标注提供短码的层级.
    """
    if LANGUAGE != "auto":
        return LANGUAGE, "LANGUAGE"

    locale = locale_settings.read_comfy_locale()
    if isinstance(locale, str):
        return _LOCALE_TO_LANGUAGE.get(locale, _DEFAULT_LANGUAGE), "Comfy.Locale"

    frontend_locale = locale_settings.get_language_setting("frontend_locale")
    if isinstance(frontend_locale, str):
        return _LOCALE_TO_LANGUAGE.get(frontend_locale, _DEFAULT_LANGUAGE), "frontend_locale"

    return _DEFAULT_LANGUAGE, "default"


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


_lang_code, _lang_source = _resolve_language()
LANG = _load_language(_lang_code)
# 语言文件缺失回退时本日志仍报解析目标, 实际加载语言由上方回退 warning 说明
logging.getLogger("llama-cpp-vulkan").info(
    _RESOLVED_INFO.get(_lang_code, _RESOLVED_INFO[_DEFAULT_LANGUAGE]).format(lang=_lang_code, source=_lang_source)
)
