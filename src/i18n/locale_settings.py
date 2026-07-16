"""语言自动跟随的设置层: ComfyUI 语言设置 (Comfy.Locale) 的实时读取, 插件根 settings.json 的读写, i18n 最底层.

settings.json 为运行时产物 (已 .gitignore), 结构: {"language": {"comfy_locale": ..., "frontend_locale": ...}}.
- comfy_locale: 每次启动实时读到的 Comfy.Locale 原始短码 (忠实记录, 不参与解析, 键缺失时移除);
- frontend_locale: 前端页面加载时上报的实际显示语言 (Comfy.Locale 缺失时的解析兜底).
两键均存前端原始短码 (zh / en / ja 等), 短码到语言文件的映射统一在 lang.py 读取时做.
读写失败静默容忍 (文件缺失/损坏按空字典处理), 持久化状态丢失只损失兜底, 不阻断插件加载.
"""

import contextlib
import json
from pathlib import Path

import folder_paths

_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.json"


def read_comfy_locale():
    """实时读 ComfyUI 用户设置的 Comfy.Locale 并落盘到 language.comfy_locale, 返回原始短码 (缺失时 None).

    非 --multi-user 模式下用户固定为 "default" (ComfyUI app/user_manager.py);
    设置文件缺失 (用户从未保存过设置) 或损坏 (ComfyUI 自身会报告) 视同键缺失.
    落盘是忠实记录 (键缺失时移除), 语言解析用返回的实时值, 不消费落盘副本.
    """
    settings_path = Path(folder_paths.get_user_directory()) / "default" / "comfy.settings.json"
    try:
        locale = json.loads(settings_path.read_text(encoding="utf-8")).get("Comfy.Locale")
    except (OSError, ValueError):
        locale = None
    set_language_setting("comfy_locale", locale)
    return locale


def _read_settings():
    try:
        settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) else {}


def get_language_setting(key):
    """返回 language 分组下 key 的值, 缺失时 None."""
    language = _read_settings().get("language")
    return language.get(key) if isinstance(language, dict) else None


def set_language_setting(key, value):
    """写入 language 分组下的 key (保留文件中其余内容); value 为 None 时移除该键, 内容无变化时不落盘."""
    settings = _read_settings()
    language = settings.get("language")
    language = language if isinstance(language, dict) else {}
    if value is None:
        if key not in language:
            return
        new_language = {k: v for k, v in language.items() if k != key}
    else:
        if language.get(key) == value:
            return
        new_language = {**language, key: value}
    with contextlib.suppress(OSError):
        _SETTINGS_PATH.write_text(json.dumps({**settings, "language": new_language}, indent=4) + "\n", encoding="utf-8")
