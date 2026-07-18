"""前端语言上报路由: 接收 web/locale_sync.js 页面加载时上报的实际显示语言, 持久化供下次启动的语言解析兜底.

Comfy.Locale 从未设置时前端按浏览器语言自动检测, 服务端无从得知实际显示
语言, 本路由把它记录到插件根 settings.json 的 language.frontend_locale
(消费方与三级优先级见 i18n/lang.py). 上报到达时本次启动的文案已固化,
生效恒为下次启动. 路由装饰器在本模块 import 时注册 (根入口 import 触发),
早于 ComfyUI 挂载路由 (main.py 先 init_extra_nodes 后 add_routes).
JS 失效只损失下次启动的兜底, 语言解析的第 1 级与默认回退不受影响.
"""

from aiohttp import web
from server import PromptServer

from ..i18n.common_static import LOG_PREFIX
from ..i18n.lang import LANG
from ..i18n.locale_settings import set_language_setting
from ..shared.logger import logger

_LOGS = LANG["logs"]["locale_sync"]

# 前端短码最长形如 zh-TW, 限长拦截异常提交, 不做短码名单校验
# (未映射短码由 lang.py 读取时落默认英语, 名单只需一处)
_MAX_LOCALE_LENGTH = 16


@PromptServer.instance.routes.post("/llama_cpp_vulkan/frontend_locale")
async def _set_frontend_locale(request):
    try:
        payload = await request.json()
    except (ValueError, LookupError):
        # LookupError: json() 按 Content-Type 的 charset 解码请求体, 伪造未知 charset 时抛出
        return web.Response(status=400)
    locale = payload.get("locale") if isinstance(payload, dict) else None
    if not isinstance(locale, str) or not locale or len(locale) > _MAX_LOCALE_LENGTH:
        return web.Response(status=400)
    set_language_setting("frontend_locale", locale)
    logger.debug(LOG_PREFIX + _LOGS["frontend_locale_saved"].format(locale=locale))
    return web.Response(status=200)
