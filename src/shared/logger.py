"""插件统一日志器, 沿用 ComfyUI 的 logging 配置(级别标签由其 Formatter 添加).

消息统一带 "[llama-cpp-vulkan]" 前缀, 便于在服务器日志中过滤.
"""

import logging

logger = logging.getLogger("llama-cpp-vulkan")
