"""插件统一日志器, 沿用 ComfyUI 的 logging 配置(级别标签由其 Formatter 添加).

消息统一带 "[llama-cpp-vulkan]" 前缀, 便于在服务器日志中过滤;
节点执行日志再叠加 "[节点名]" 前缀 (经 node_log_prefix 构造), 定位日志来源节点.
"""

import logging

from ..i18n.common_static import LOG_PREFIX

logger = logging.getLogger("llama-cpp-vulkan")


def node_log_prefix(node_name):
    """节点日志前缀 "[llama-cpp-vulkan] [节点名] "; 节点名是功能性标识, 不进 i18n."""
    return f"{LOG_PREFIX}[{node_name}] "
