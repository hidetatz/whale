from enum import IntEnum, auto

class Ops(IntEnum):
    Const = auto()

    _unary_start = auto()
    Neg = auto()
    Pow = auto()
    Sin = auto()
    Cos = auto()
    Exp = auto()
    Log = auto()
    Sqrt = auto()
    _unary_end = auto()

    _binary_start = auto()
    Add = auto()
    Sub = auto()
    Mul = auto()
    Truediv = auto()
    _binary_end = auto()

    _reduce_start = auto()
    Sum = auto()
    _reduce_end = auto()

    _view_start = auto()
    Reshape = auto()
    Broadcast = auto()
    Slice = auto()
    Transpose = auto()
    _view_end = auto()

    Matmul = auto()

    def is_const(self): return self.value == Ops.Const
    def is_unary(self): return Ops._unary_start < self.value < Ops._unary_end
    def is_binary(self): return Ops._binary_start < self.value < Ops._binary_end
    def is_reduce(self): return Ops._reduce_start < self.value < Ops._reduce_end
    def is_view(self): return Ops._view_start < self.value < Ops._view_end
    def is_matmul(self): return self.value == Ops.Matmul

    def __repr__(self): return self.name
    def __str__(self): return self.name
