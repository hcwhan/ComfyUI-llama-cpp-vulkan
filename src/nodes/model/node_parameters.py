"""采样参数配置节点, 打包成 kwargs dict 透传给 create_chat_completion."""

from ...core.instruct import DEFAULT_SAMPLING_PARAMS as _DEFAULTS


class llama_cpp_parameters:
    @classmethod
    def INPUT_TYPES(s):
        # widget 默认值与 Instruct 未连接 parameters 端口时的生效值同源
        # (core/instruct.py 的 DEFAULT_SAMPLING_PARAMS), 保证两种接法行为一致
        return {
            "required": {
                "max_tokens": ("INT", {"default": _DEFAULTS["max_tokens"], "min": 0, "max": 65536, "step": 1, "tooltip": f"生成 token 数上限, 0 = 不限制 (受 n_ctx 约束).\n默认 {_DEFAULTS['max_tokens']}."}),
                "top_k": ("INT", {"default": _DEFAULTS["top_k"], "min": 0, "max": 1000, "step": 1, "tooltip": f"只保留概率最高的 K 个候选 token 再采样, 0 = 不裁剪.\n默认 {_DEFAULTS['top_k']}."}),
                "top_p": ("FLOAT", {"default": _DEFAULTS["top_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": f"核采样: 按概率从高到低累计到 p 截断候选集, 1.0 = 禁用.\n默认 {_DEFAULTS['top_p']}."}),
                "min_p": ("FLOAT", {"default": _DEFAULTS["min_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": f"相对概率下限: 剔除低于 最高候选概率 x min_p 的候选, 0 = 禁用.\n默认 {_DEFAULTS['min_p']}."}),
                "typical_p": ("FLOAT", {"default": _DEFAULTS["typical_p"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": f"典型性采样: 按信息量偏离度筛选候选, 1.0 = 禁用.\n默认 {_DEFAULTS['typical_p']}."}),
                "temperature": ("FLOAT", {"default": _DEFAULTS["temperature"], "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": f"采样温度: 越低越确定保守, 越高越发散随机, 0 = 贪心.\n默认 {_DEFAULTS['temperature']}."}),
                "repeat_penalty": ("FLOAT", {"default": _DEFAULTS["repeat_penalty"], "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": f"对最近窗口内出现过的 token 乘惩罚系数, 1.0 = 禁用.\n默认 {_DEFAULTS['repeat_penalty']}."}),
                # OpenAI/llama.cpp 语义均允许负值(奖励重复), 统一 -2.0 ~ 2.0
                "frequency_penalty": ("FLOAT", {"default": _DEFAULTS["frequency_penalty"], "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": f"按 token 已出现的次数线性累加惩罚, 负值奖励重复.\n默认 {_DEFAULTS['frequency_penalty']}."}),
                "present_penalty": ("FLOAT", {"default": _DEFAULTS["present_penalty"], "min": -2.0, "max": 2.0, "step": 0.01, "tooltip": f"token 只要出现过就惩罚一次, 鼓励引入新内容, 负值奖励重复.\n(即 OpenAI 的 presence_penalty)\n默认 {_DEFAULTS['present_penalty']}."}),
                "mirostat_mode": ("INT", {"default": _DEFAULTS["mirostat_mode"], "min": 0, "max": 2, "step": 1, "tooltip": f"Mirostat 自适应采样: 0 = 关闭, 1 = Mirostat, 2 = Mirostat 2.0.\n开启后接管采样, top_k/top_p 等失效.\n默认 {_DEFAULTS['mirostat_mode']}."}),
                "mirostat_eta": ("FLOAT", {"default": _DEFAULTS["mirostat_eta"], "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": f"Mirostat 学习率: 控制向目标熵收敛的速度.\n默认 {_DEFAULTS['mirostat_eta']}."}),
                "mirostat_tau": ("FLOAT", {"default": _DEFAULTS["mirostat_tau"], "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": f"Mirostat 目标熵 (约为目标困惑度), 越大输出越多样.\n默认 {_DEFAULTS['mirostat_tau']}."}),
            }
        }

    RETURN_TYPES = ("LLAMACPPARAMS",)
    RETURN_NAMES = ("parameters",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vulkan"

    def process(self, **kwargs):
        return (kwargs,)
