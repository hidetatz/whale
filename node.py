import operator
from functools import reduce

from buffer import Buffer, CPUBuff, DevBuff

class Node:
    def __init__(self, dtype, shape, strides, offset, val=None, ctx=None, buffer=None):
        if buffer is not None:
            self.buffer = buffer
        else:
            if val is not None:
                self.buffer = Buffer(dtype=dtype, length=len(val), cpu=CPUBuff(val), dev=DevBuff())
            else:
                length = reduce(operator.mul, shape, 1)
                self.buffer = Buffer(dtype=dtype, length=length, cpu=None, dev=DevBuff())
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.ctx = ctx
