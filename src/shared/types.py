"""AnyType 万能透传类型, 令 ComfyUI 类型校验对 "*" 端口放行."""


class AnyType(str):
    """重载 __ne__ 恒等 False, 使 "*" 端口与任意类型双向放行 (任意类型可连入 "*" 输入, "*" 输出可连任意输入)."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")
