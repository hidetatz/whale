from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, auto

import cuda
import device


def shape_to_strides(shape: list[int]):
    return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])


class TensorOpCode(IntEnum):
    _buffer_op_start = auto()
    BUFFER = auto()
    _buffer_op_end = auto()

    _unary_op_start = auto()
    RECIP = auto()
    _unary_op_end = auto()

    _binary_op_start = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()
    _binary_op_end = auto()

    def _in(self, start, end):
        return start < self.value and self.value < end

    def is_buffer_op(self):
        return self._in(TensorOpCode._buffer_op_start, TensorOpCode._buffer_op_end)

    def is_unary_op(self):
        return self._in(TensorOpCode._unary_op_start, TensorOpCode._unary_op_end)

    def is_binary_op(self):
        return self._in(TensorOpCode._binary_op_start, TensorOpCode._binary_op_end)

    def __str__(self):
        return self.name


cuda_device = cuda.Device()


def get_device():
    return cuda_device


@dataclass
class TensorOp:
    code: TensorOpCode
    shape: tuple[int]
    strides: tuple[int]
    offset: int
    inputs: tuple[TensorOp]
    cpu_buffer: device.CPUMemoryBuffer
    dev_buffer: device.DeviceMemoryBuffer
    differentiable: Differentiable
    generation: int
    dev: device.Device

    @classmethod
    def new_buffer_op(cls, data: any, shape: tuple(int) = None):
        code = TensorOpCode.BUFFER

        # scalar
        if type(data) == float:
            return TensorOp(code, (), (), 0, (), device.CPUMemoryBuffer([data]), None, None, 0, get_device())

        if type(data) == int:
            return TensorOp(code, (), (), 0, (), device.CPUMemoryBuffer([float(data)]), None, None, 0, get_device())

        if shape is not None:
            return TensorOp(code, shape, shape_to_strides(shape), 0, (), device.CPUMemoryBuffer(data), None, None, 0, get_device())

        # tensor
        if type(data) == list:
            flattened = []
            shape = []

            def f(d, dim):
                if type(d) != int and type(d) != float and type(d) != list:
                    raise ValueError(f"array must be a multi-dimensional array of int or float: {data}")

                if type(d) == int or type(d) == float:
                    flattened.append(d if type(d) == float else float(d))
                    return

                # d must be list here
                length = len(d)
                if len(shape) == dim:
                    shape.append(length)
                elif length != shape[dim]:
                    raise ValueError(f"array must be homogeneous: {data}")

                for elem in d:
                    f(elem, dim + 1)

            f(data, 0)
            return TensorOp(code, tuple(shape), shape_to_strides(shape), 0, (), device.CPUMemoryBuffer(flattened), None, None, 0, get_device())

        raise TypeError(f"type {type(data)} is unsupported as Tensor")

    @classmethod
    def new_binary_op(cls, d: DifferentiableBinaryCalc, l: TensorOp, r: TensorOp):
        return d.forward((l, r))
        # # t.creator = d
        # # t.generation = calc.generation + 1
        # return t

    @classmethod
    def new_unary_op(cls, d: DifferentiableUnaryCalc, src: TensorOp):
        return d.forward((src,))

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def recip(self):
        return TensorOp.new_unary_op(Recip(), self)

    def __add__(self, r: TensorOp):
        return TensorOp.new_binary_op(Add(), self, r)

    def __mul__(self, r: TensorOp):
        return TensorOp.new_binary_op(Mul(), self, r)

    def __pow__(self, r: TensorOp):
        return TensorOp.new_binary_op(Pow(), self, r)

    def __str__(self):
        def f(depth: int, op: TensorOp):
            indent = "    " * depth
            trail_comma = "," if depth != 0 else ""

            if not op.inputs:
                return f"{indent}{op.str_oneline()}{trail_comma}"

            inputs = "[\n" + "\n".join([f(depth + 1, i) for i in op.inputs]) + f"\n{indent}]"
            return f"{indent}{op.str_oneline().rstrip(')')} inputs: {inputs}{trail_comma}"

        return f(0, self)

    def str_oneline(self):
        return f"TensorOp(code: {self.code} {self.shape}_{self.strides}_{self.offset} py_buff:{'o' if self.cpu_buffer else 'x'} dev_buff:{'o' if self.dev_buffer else 'x'})"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        # todo: free the device buffer
        pass


class Differentiable:
    def forward(self, inputs: tuple(TensorOp)):
        self.inputs = inputs
        self.generation = max([i.generation for i in inputs])
        self.output = self._forward(*inputs)
        return self.output

    def backward(self, grad):
        return self._backward(grad)

    def _forward(self, inputs):
        raise NotImplementedError()

    def _backward(self, grad):
        raise NotImplementedError()


# unary
class DifferentiableUnaryCalc(Differentiable):
    def _forward(self, op: TensorOp) -> TensorOp:
        self.op = op
        return TensorOp(self._forward_code(), op.shape, shape_to_strides(op.shape), 0, (op,), None, None, self, op.generation, get_device())

    def _backward(self, grad: TensorOp) -> tuple(TensorOp):
        raise NotImplementedError()


class Recip(DifferentiableUnaryCalc):
    def _forward_code(self):
        return TensorOpCode.RECIP

    def _backward(self, grad):
        return grad * (full(self.inputs[0].shape, -1.0) / (self.inputs[0] * self.inputs[0]))


# binary
class DifferentiableBinaryCalc(Differentiable):
    def _forward(self, l: TensorOp, r: TensorOp) -> TensorOp:
        self.l = l
        self.r = r
        return TensorOp(
            self._forward_code(), l.shape, shape_to_strides(l.shape), 0, (l, r), None, None, self, max(l.generation, r.generation), get_device()
        )

    def _backward(self, grad: TensorOp) -> tuple(TensorOp):
        raise NotImplementedError()


class Add(DifferentiableBinaryCalc):
    def _forward_code(self):
        return TensorOpCode.ADD

    def _backward(self, grad: TensorOp):
        return grad, grad


class Mul(DifferentiableBinaryCalc):
    def _forward_code(self):
        return TensorOpCode.MUL

    def _backward(self, grad: TensorOp):
        return grad * self.r, grad * self.l


class Pow(DifferentiableBinaryCalc):
    def _forward_code(self):
        return TensorOpCode.POW

    def _backward(self, grad):
        raise NotImplementedError()


# class GetItem(BackpropContext):
#     def __init__(self, newshape, newstrides, newoffset):
#         self.newshape = newshape
#         self.newstrides = newstrides
#         self.newoffset = newoffset

#     def _forward(self, inputs):
#         return Tensor(
#             shape=self.newshape,
#             strides=self.newstrides,
#             offset=self.newoffset,
#             op=TensorOp.GETITEM,
#             view=True,
#             materialized=True,
#             inputs=inputs,
#             dtype=inputs[0].dtype,
#         )

#     def _backward(self, grad):
#         raise NotImplementedError()
