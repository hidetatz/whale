from functools import reduce
import operator

import backend
from buffer import Buffer, CPUBuff, DevBuff

class Node:
    def __init__(self, val, dtype, shape, strides, offset, ctx):
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

    def __del__(self):
        if self.buffer.dev.ptr: backend.free(self.buffer.dev.ptr)

