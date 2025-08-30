from __future__ import annotations


class DType:
    def __init__(self, code: str) -> None:
        self.code = code

    def __eq__(self, r) -> bool:
        return self.code == r.code

    def __str__(self) -> str:
        return "dtypes." + self.code

    @staticmethod
    def float32() -> DType:
        return DType("float32")

    @staticmethod
    def bool() -> DType:
        return DType("bool")


class dtypes:
    float32 = DType.float32()
    bool = DType.bool()
