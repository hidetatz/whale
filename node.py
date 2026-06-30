from functools import reduce
import operator

from buffer import Buffer, CPUBuff

class Node:
    def __init__(self, val, dtype, shape, strides, offset, ctx):
        if val:
            self.buffer = Buffer(cpu=CPUBuff(val, dtype), dev=None)
        else:
            self.buffer = Buffer(cpu=CPUBuff([0] * reduce(operator.mul, shape, 1), dtype), dev=None)
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.ctx = ctx

