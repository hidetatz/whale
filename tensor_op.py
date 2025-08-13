from enum import Enum, auto


class TensorOp(Enum):
    CONST = auto()

    # arithmetic
    ADD = auto()
    MUL = auto()
    RECIP = auto()
    POW = auto()
    MATMUL = auto()

    MAXIMUM = auto()

    # movement
    GETITEM = auto()
