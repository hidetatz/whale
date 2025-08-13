import collections
from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import time

import backend
import device
import materialize
from tensor_op import TensorOp

class BackpropContext:
    def forward(self, inputs):
        self.inputs = inputs
        self.generation = max([i.generation for i in inputs])
        self.output = self._forward(inputs)
        return self.output

    def backward(self, grad):
        return self._backward(grad)

    def _forward(self, inputs):
        raise NotImplementedError()

    def _backward(self, grad):
        raise NotImplementedError()


class Add(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape,
            strides=inputs[0].strides,
            offset=inputs[0].offset,
            op=TensorOp.ADD,
            inputs=inputs,
            dtype=inputs[0].dtype,
        )

    def _backward(self, grad):
        return grad, grad


class Mul(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape,
            strides=inputs[0].strides,
            offset=inputs[0].offset,
            op=TensorOp.MUL,
            inputs=inputs,
            dtype=inputs[0].dtype,
        )

    def _backward(self, grad):
        return grad * self.inputs[1], grad * self.inputs[0]


class Recip(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape,
            strides=inputs[0].strides,
            offset=inputs[0].offset,
            op=TensorOp.RECIP,
            inputs=inputs,
            dtype=inputs[0].dtype,
        )

    def _backward(self, grad):
        return grad * (full(self.inputs[0].shape, -1.0) / (self.inputs[0] * self.inputs[0]))


class Pow(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape,
            strides=inputs[0].strides,
            offset=inputs[0].offset,
            op=TensorOp.POW,
            inputs=inputs,
            dtype=inputs[0].dtype,
        )

    def _backward(self, grad):
        raise NotImplementedError()


class GetItem(BackpropContext):
    def __init__(self, newshape, newstrides, newoffset):
        self.newshape = newshape
        self.newstrides = newstrides
        self.newoffset = newoffset

    def _forward(self, inputs):
        return Tensor(
            shape=self.newshape,
            strides=self.newstrides,
            offset=self.newoffset,
            op=TensorOp.GETITEM,
            view=True,
            inputs=inputs,
            dtype=inputs[0].dtype,
        )

    def _backward(self, grad):
        raise NotImplementedError()


class DType(Enum):
    I32 = auto()
    F32 = auto()

    @classmethod
    def from_type(cls, typ):
        if typ == int:
            return cls.I32
        if typ == float:
            return cls.F32


class Tensor:
    def __init__(self, data=[], shape=[], strides=[], offset=0, op=None, creator=None, inputs=[], materialized=False, view=False, dtype=None):
        # actual data on python
        self.python_buffer: device.PythonBuffer = device.PythonBuffer(data if data else None)

        # actual data on gpu. Loaded when needed
        self.device_buffer: device.GPUBuffer = None

        # data type
        self.dtype: DType = dtype

        # dimensions
        self.shape: list[int] = shape
        self.strides: list[int] = strides
        self.offset: int = offset

        # tensor operation
        self.op: TensorOp = op
        self.inputs: list[Tensor] = inputs

        # backprop context
        self.grad: Tensor = None
        self.generation: int = 0
        self.creator:BackpropContext = creator
        self.materialized: bool = materialized

        # view tensor params
        self.view: bool = view

    @property
    def data(self):
        return self.base.python_buffer.value

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def base(self):
        return self.inputs[0].base if self.view else self

    def materialize(self):
        if self.view:
            self.base.materialize()
            return

        if not self.materialized:
            materialize.Materializer.materialize(self)
            self.materialized = True

    def add(self, r):
        return _from_calc(Add(), [self, r])

    def sub(self, r):
        # l-r is l+(-1*r)
        return _from_calc(Add(), [self, _from_calc(Mul(), [full(r.shape, -1.0), r])])

    def mul(self, r):
        return _from_calc(Mul(), [self, r])

    def div(self, r):
        # l/r = l * (1/r)
        return _from_calc(Mul(), [self, _from_calc(Recip(), [r])])

    def pow(self, r):
        return _from_calc(Pow(), [self, r])

    def _getitem(self, indices):
        if type(indices) != tuple:
            indices = (indices,)

        if self._is_scalar():
            raise ValueError(f"cannot index on scalar: {self}")

        if self.ndim < len(indices):
            raise ValueError(f"too many index accessors specified")

        if Tensor in [type(i) for i in indices]:
            return self._get_item__advanced(indices)

        return self._get_item_basic(indices)

    # numpy basic index
    # https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
    def _get_item_basic(self, indices):
        for i in range(self.ndim - len(indices)):
            indices = indices + (slice(None, None, None),)

        newshape = [0] * len(self.shape)
        newstrides = [0] * len(self.strides)
        newoffset = self.offset

        for i, idx in enumerate(indices):
            if type(idx) == int:
                if self.shape[i] - 1 < idx:
                    raise ValueError(f"index out of bounds for axis {i} with size {self.shape[i]}")

                newoffset += idx * self.strides[i]
                newshape[i] = -1  # dummy
                newstrides[i] = -1  # dummy

            elif type(idx) == slice:
                start = 0 if idx.start is None else idx.start
                stop = self.shape[i] if idx.stop is None else idx.stop
                step = 1 if idx.step is None else idx.step
                if step == 0:
                    raise ValueError(f"step in slice index must not be 0: {idx}")

                if self.shape[i] < start or stop < start:
                    newshape[i] = 0
                else:
                    newshape[i] = (stop - start + step - 1) // step

                newstrides[i] = self.strides[i] * step

                if newshape[i] != 0:
                    newoffset += start * self.strides[i]

            else:
                raise RuntimeError(f"unhandled index in basic index: {type(idx)}")

        newshape = tuple([s for s in newshape if s != -1])
        newstrides = tuple([s for s in newstrides if s != -1])
        return _from_calc(GetItem(newshape, newstrides, newoffset), [self])

    def __add__(self, r):
        return self.add(r)

    def __sub__(self, r):
        return self.sub(r)

    def __mul__(self, r):
        return self.mul(r)

    def __truediv__(self, r):
        return self.div(r)

    def __pow__(self, r):
        return self.pow(r)

    def __getitem__(self, indices):
        return self._getitem(indices)

    def __matmul__(self, r):
        return _from_calc(MATMUL, [self, r])

    def _is_scalar(self):
        return self.ndim == 0

    def backprop(self):
        if self.view:
            self.base.backprop()

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

    def tolist(self):
        # todo: fix
        return self.base.data

    def str_as_ndarr(self):
        if len(self.shape) == 0:
            return f"Tensor({self.data[0]})"

        if 0 in self.shape:
            return f"Tensor([])"

        if len(self.shape) == 1:
            return f"Tensor([{', '.join(map(str, self.data))}])"

        # todo: implement
        return f"Tensor(data: {self.data}, offset: {self.offset}, shape: {self.shape}, strides: {self.strides})"

    def str_as_graph(self):
        def f(depth: int, t: Tensor):
            indent = "    " * depth
            trail_comma = "," if depth != 0 else ""

            if not t.inputs:
                return f"{indent}Tensor(view: False, op:{t.op}, offset: {t.offset}, shape:{t.shape}, strides: {t.strides}){trail_comma}"

            input = "[\n" + "\n".join([f(depth + 1, i) for i in t.inputs]) + f"\n{indent}]"
            return f"{indent}Tensor(view: {t.view}, op:{t.op}, offset: {t.offset}, shape:{t.shape}, strides: {t.strides}, input: {input}){trail_comma}"

        return f(0, self)

    def str_as_oneline(self):
        return f"Tensor(op:{self.op}, shape:{self.shape}, strides:{self.strides} offset:{self.offset})"

    def __str__(self):
        return self.str_as_ndarr() if self.materialized or self.view else self.str_as_graph()

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        if self.device_buffer:
            # todo: free device_buffer
            pass


def _from_calc(calc, inputs):
    t = calc.forward(inputs)
    t.creator = calc
    t.generation = calc.generation + 1
    return t


def tensor_with_shape(arr, shape):
    strides = tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])
    return Tensor(
        data=arr,
        shape=shape,
        strides=strides,
        offset=0,
        op=TensorOp.CONST,
        materialized=True,
        dtype=DType.from_type(type(arr[0])),
    )


def array(arr):
    if type(arr) == int or type(arr) == float:
        return Tensor(data=[arr], op=TensorOp.CONST, materialized=True, dtype=DType.from_type(type(arr)))

    elif type(arr) == list:
        data = []
        shape = []

        def f(d, dim):
            if type(d) != int and type(d) != float and type(d) != list:
                raise ValueError(f"array must be a multi-dimensional array of int or float: {arr}")

            if type(d) == int or type(d) == float:
                data.append(d)
                return

            # d must be list here
            length = len(d)
            if len(shape) == dim:
                shape.append(length)
            else:
                if length != shape[dim]:
                    raise ValueError(f"array must be homogeneous: {arr}")

            for elem in d:
                f(elem, dim + 1)

        f(arr, 0)
        return tensor_with_shape(data, tuple(shape))


def full(shape, value):
    return tensor_with_shape([value] * math.prod(shape), shape)


def ones_like(t):
    return full(t.shape, 1.0)


def zeros_like(t):
    return full(t.shape, 0.0)


if __name__ == "__main__":
    # t1 = array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    # t2 = array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    # t3 = t1[0, 1]
    # t4 = t2[1, 2]
    # t5 = t3 + t4
    # t5.materialize()
    # print(t5)
    # t2.materialize()
    # print(t2.tolist())
    t1 = array([[1, 2, 3], [1, 2, 3]])
    t2 = t1[0]
    print(t2)
    # t2 = array([[4, 5, 6], [4, 5, 6]])
    # t3 = t1 + t2
    # t4 = array([[7, 8, 9], [7, 8, 9]])
    # t5 = t3 * t4
    # t5.materialize()
    # print(t5)

    # t5.backprop()
    # t1.grad.materialize()
    # print(t1.grad)
