"""AnyType 万能透传类型, 令 ComfyUI 类型校验对 "*" 端口放行."""

class AnyType(str):
    """重载 __ne__ 恒等 False, 使任意类型都能连入声明为 "*" 的端口."""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")
