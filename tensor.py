from __future__ import annotations

import collections
import itertools
import math
import os
import time
from dataclasses import dataclass
from enum import Enum, auto

import backend
import cuda
import device
import materialize
from tensor_op import TensorOp


class Tensor:
    #
    # constructors
    #

    def __init__(self, arg: any):
        self.grad: Tensor = None
        self.materialized = False

        if type(arg) == int or type(arg) == float or type(arg) == list:
            self.op = TensorOp.new_buffer_op(arg)
            self.materialized = True

        elif type(arg) == TensorOp:
            self.op = arg

        else:
            raise TypeError(f"cannot handle {type(arg)} to initialize Tensor")

    @classmethod
    def full(cls, shape: tuple[int], val: float):
        return Tensor(TensorOp.new_buffer_op([val] * math.prod(shape), shape=shape))

    @classmethod
    def full_like(cls, t: Tensor, val: float):
        return Tensor.full(t.shape, val)

    #
    # properties
    #

    @property
    def shape(self):
        return self.op.shape

    @property
    def strides(self):
        return self.op.strides

    @property
    def offset(self):
        return self.op.offset

    @property
    def size(self):
        return self.op.size

    @property
    def ndim(self):
        return self.op.ndim

    def _is_scalar(self):
        return self.ndim == 0

    #
    # operators
    #

    def add(self, r: Tensor):
        return Tensor(self.op + r.op)

    def sub(self, r: Tensor):
        # l-r = l + (r*-1)
        return self + (r * Tensor.full_like(r, -1))

    def mul(self, r: Tensor):
        return Tensor(self.op * r.op)

    def truediv(self, r: Tensor):
        # l/r = l * (1/r)
        return self.op * r.recip()

    def recip(self):
        return Tensor(self.op.recip())

    def pow(self, r: Tensor):
        return Tensor(self.op**r.op)

    # def _getitem(self, indices):
    #     if type(indices) != tuple:
    #         indices = (indices,)

    #     if self._is_scalar():
    #         raise ValueError(f"cannot index on scalar: {self}")

    #     if self.ndim < len(indices):
    #         raise ValueError(f"too many index accessors specified")

    #     if Tensor in [type(i) for i in indices]:
    #         return self._get_item__advanced(indices)

    #     return self._get_item_basic(indices)

    # # numpy basic index
    # # https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
    # def _get_item_basic(self, indices):
    #     for i in range(self.ndim - len(indices)):
    #         indices = indices + (slice(None, None, None),)

    #     newshape = [0] * len(self.shape)
    #     newstrides = [0] * len(self.strides)
    #     newoffset = self.offset

    #     for i, idx in enumerate(indices):
    #         if type(idx) == int:
    #             if self.shape[i] - 1 < idx:
    #                 raise ValueError(f"index out of bounds for axis {i} with size {self.shape[i]}")

    #             newoffset += idx * self.strides[i]
    #             newshape[i] = -1  # dummy
    #             newstrides[i] = -1  # dummy

    #         elif type(idx) == slice:
    #             start = 0 if idx.start is None else idx.start
    #             stop = self.shape[i] if idx.stop is None else idx.stop
    #             step = 1 if idx.step is None else idx.step
    #             if step == 0:
    #                 raise ValueError(f"step in slice index must not be 0: {idx}")

    #             if self.shape[i] < start or stop < start:
    #                 newshape[i] = 0
    #             else:
    #                 newshape[i] = (stop - start + step - 1) // step

    #             newstrides[i] = self.strides[i] * step

    #             if newshape[i] != 0:
    #                 newoffset += start * self.strides[i]

    #         else:
    #             raise RuntimeError(f"unhandled index in basic index: {type(idx)}")

    #     newshape = tuple([s for s in newshape if s != -1])
    #     newstrides = tuple([s for s in newstrides if s != -1])
    #     return _from_calc(GetItem(newshape, newstrides, newoffset), [self])

    def __add__(self, r):
        return self.add(r)

    def __sub__(self, r):
        return self.sub(r)

    def __mul__(self, r):
        return self.mul(r)

    def __truediv__(self, r):
        return self.truediv(r)

    def __pow__(self, r):
        return self.pow(r)

    # def __getitem__(self, indices):
    #     return self._getitem(indices)

    def __str__(self):
        return f"Tensor({self.op})"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        self.op.__del__()

    #
    # materialization
    #

    def materialize(self):
        if not self.materialized:
            materialize.Materializer.materialize(self.op)
            self.materialized = True

    def tolist(self):
        if not self.materialized:
            self.materialize()

        if self.op.cpu_buffer is None:
            self.op.cpu_buffer = device.CPUMemoryBuffer(None)
        self.op.dev.copy_from_device(self.op.dev_buffer, self.op.cpu_buffer)

        # calculate index combination by shape
        indices = [list(c) for c in list(itertools.product(*[range(n) for n in self.shape]))]

        # pick the actual value to be in the result by the index
        vals = []
        for idx in indices:
            vals.append(self.op.cpu_buffer.raw[self.offset + sum([st * i for st, i in zip(self.strides, idx)])])

        # early return on scalar
        if len(self.shape) == 0:
            return vals[0]

        # do reshape manually
        def _reshape(data, dims, start):
            if len(dims) == 1:
                return data[start : start + dims[0]]
            else:
                result = []
                curdim = dims[0]
                remaindims = dims[1:]
                elems = math.prod(remaindims)

                for i in range(curdim):
                    result.append(_reshape(data, remaindims, start + i * elems))

                return result

        return _reshape(vals, self.shape, 0)

    #
    # back propagation
    #

    def backprop(self):
        if self.grad is None:
            self.grad = ones_like(self)

        calcs = []
        seen = set()

        def f(c):
            if c not in seen:
                calcs.append(c)
                seen.add(c)
                calcs.sort(key=lambda x: x.generation)

        f(self.creator)

        while calcs:
            c = calcs.pop()
            gy = c.output.grad
            gxs = c.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(c.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    f(x.creator)

    def backward(self):
        self.backprop()


if __name__ == "__main__":
    # t1 = array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    # t2 = array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    # t3 = t1[0, 1]
    # t4 = t2[1, 2]
    # t5 = t3 + t4
    # print(t3.tolist(), t4.tolist())
    # print(t5)
    # t5.materialize()
    # print(t5)
    # t2.materialize()
    # print(t2.tolist())
    t1 = Tensor([[1, 2, 3], [1, 2, 3]])
    t2 = Tensor([[4, 5, 6], [4, 5, 6]])
    t3 = t1 + t2
    t3.materialize()
    print(t3.tolist())
    # print(t1)
    # t2 = t1[0, 1]
    # print(t2)
    # print(type(t2.tolist()))
    # t2 = Tensor([[4, 5, 6], [4, 5, 6]])
    # t3 = t1 + t2
    # # t4 = array([[7, 8, 9], [7, 8, 9]])
    # # t5 = t3 * t4
    # t3.materialize()
    # print(t3)

    # t5.backprop()
    # t1.grad.materialize()
    # print(t1.grad)
