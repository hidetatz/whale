from __future__ import annotations

import collections
import itertools
import math
import os
import time
import typing
from dataclasses import dataclass
from enum import IntEnum, auto

import cuda
import device
import kernel
from backend import Backend
from dtype import DType, dtypes

dbg = os.getenv("WHALE_DEBUG", "") != ""

backend = Backend.detect()

cuda_device = cuda.Device()


def get_device():
    if backend == Backend.CUDA:
        return cuda_device

    raise RuntimeError("no backend")


def shape_to_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])


class TensorOpCode(IntEnum):
    _buffer_op_start = auto()
    BUFFER = auto()
    _buffer_op_end = auto()

    _unary_op_start = auto()
    RECIP = auto()
    LOG = auto()
    SIN = auto()
    COS = auto()
    TANH = auto()
    EXP = auto()
    COPY = auto()
    _unary_op_end = auto()

    _binary_op_start = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()
    NE = auto()
    LT = auto()
    MAXIMUM = auto()
    _binary_op_end = auto()

    _reduce_op_start = auto()
    SUM = auto()
    PROD = auto()
    MAX = auto()
    _reduce_op_end = auto()

    _view_op_start = auto()
    PERMUTE = auto()
    EXPAND = auto()
    CROP = auto()
    PAD = auto()
    RESHAPE = auto()
    _view_op_end = auto()

    def _in(self, start: TensorOpCode, end: TensorOpCode) -> bool:
        return start < self.value and self.value < end

    def is_buffer_op(self) -> bool:
        return self._in(TensorOpCode._buffer_op_start, TensorOpCode._buffer_op_end)

    def is_unary_op(self) -> bool:
        return self._in(TensorOpCode._unary_op_start, TensorOpCode._unary_op_end)

    def is_binary_op(self) -> bool:
        return self._in(TensorOpCode._binary_op_start, TensorOpCode._binary_op_end)

    def is_reduce_op(self) -> bool:
        return self._in(TensorOpCode._reduce_op_start, TensorOpCode._reduce_op_end)

    def is_view_op(self) -> bool:
        return self._in(TensorOpCode._view_op_start, TensorOpCode._view_op_end)

    def __str__(self):
        return self.name


class Differentiable:
    def forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.inputs = inputs
        self.output = self._forward(inputs)
        return self.output

    def backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return self._backward(grad)

    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError()


# data
class DifferentiableData(Differentiable):
    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.src = inputs[0]
        return Tensor(self._forward_code(), shape=self.src.shape, inputs=(self.src,), backprop_ctx=self, dtype=self.src.dtype)

    def _forward_code(self) -> TensorOpCode:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad,)


class Copy(DifferentiableData):
    def _forward_code(self):
        return TensorOpCode.COPY


# unary
class DifferentiableUnary(Differentiable):
    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.src = inputs[0]
        return Tensor(self._forward_code(), shape=self.src.shape, inputs=(self.src,), backprop_ctx=self, dtype=self.src.dtype)

    def _forward_code(self) -> TensorOpCode:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError()


class Recip(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.RECIP

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad * (Tensor.full_like(self.src, -1.0) / (self.src * self.src)),)


class Log(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.LOG

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad / self.src,)


class Sin(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.SIN

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad * self.src.cos(),)


class Cos(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.COS

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad * -self.src.sin(),)


class Tanh(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.TANH

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        y = self.output
        return (grad * (1 - y * y),)


class Exp(DifferentiableUnary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.EXP

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        y = self.output
        return (grad * y,)


# binary
class DifferentiableBinary(Differentiable):
    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.l = self.inputs[0]
        self.r = self.inputs[1]
        assert self.l.dtype == self.r.dtype  # todo: this should be fixed
        return Tensor(self._forward_code(), shape=self.l.shape, inputs=(self.l, self.r), backprop_ctx=self, dtype=self.l.dtype)

    def _forward_code(self) -> TensorOpCode:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError()


class Add(DifferentiableBinary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.ADD

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return grad, grad


class Mul(DifferentiableBinary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.MUL

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return grad * self.r, grad * self.l


class Pow(DifferentiableBinary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.POW

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        lgrad = grad * self.r * (self.l ** (self.r - Tensor.full_like(self.r, 1)))
        rgrad = grad * (self.l**self.r) * self.l.log()
        return lgrad, rgrad


class Ne(DifferentiableBinary):
    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.l = self.inputs[0]
        self.r = self.inputs[1]
        assert self.l.dtype == self.r.dtype  # todo: this should be fixed
        return Tensor(TensorOpCode.NE, shape=self.l.shape, inputs=(self.l, self.r), backprop_ctx=self, dtype=dtypes.bool)

    # no backward for compare


class Lt(DifferentiableBinary):
    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.l = self.inputs[0]
        self.r = self.inputs[1]
        assert self.l.dtype == self.r.dtype  # todo: this should be fixed
        return Tensor(TensorOpCode.LT, shape=self.l.shape, inputs=(self.l, self.r), backprop_ctx=self, dtype=dtypes.bool)

    # no backward for compare


class Maximum(DifferentiableBinary):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.MAXIMUM

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        # this is not considering the same value case, likely equally split is better
        l_mask = self.l > self.r
        lgrad = l_mask.to(dtypes.float32) * grad
        rgrad = l_mask.logical_not().to(dtypes.float32) * grad
        return lgrad, rgrad


# reduce
class DifferentiableReduce(Differentiable):
    def __init__(self, axis: int, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.src = inputs[0]
        newshape = tuple([s if i != self.axis else 1 for i, s in enumerate(self.src.shape)])  # updates shape[axis] to 1
        return Tensor(self._forward_code(), shape=newshape, inputs=(self.src,), backprop_ctx=self, dtype=self.src.dtype)

    def _forward_code(self) -> TensorOpCode:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError()


def _reduce_backward_restore_grad_shape(orig_shape: tuple[int, ...], t: Tensor) -> Tensor:
    lead = len(orig_shape) - len(t.shape)
    return t.reshape(*([1] * lead + list(t.shape))).broadcast_to(orig_shape)


class Sum(DifferentiableReduce):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.SUM

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (_reduce_backward_restore_grad_shape(self.src.shape, grad),)


class Prod(DifferentiableReduce):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.PROD

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        # This implementation is different with the pytorch one.
        # When the input includes 0, the 0 division occurs and the grad will be nan.
        # In PyTorch, it calcs (grad[i] * prod(*input[:i], *input[i+1:])).
        return ((_reduce_backward_restore_grad_shape(self.src.shape, grad * self.output) / self.src),)


class Max(DifferentiableReduce):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.MAX

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        orig_shape = self.src.shape

        # mask tensor where the maximum value position is 1, otherwise 0, dtype is float32
        max_mask = self.src.eq(_reduce_backward_restore_grad_shape(orig_shape, self.output)).to(dtypes.float32)

        # sum the mask tensor to count how many max value is contained on the axis
        factor = _reduce_backward_restore_grad_shape(orig_shape, max_mask.sum(axis=self.axis, keepdims=True))

        # distribute the sum to the max value pos equally by division
        return ((max_mask / factor) * _reduce_backward_restore_grad_shape(orig_shape, grad),)


# view
class DifferentiableView(Differentiable):
    def __init__(
        self, shape: tuple[int, ...], strides: tuple[int, ...], offset: int, valid_area: tuple[tuple[int, int], ...] | None, contiguous: bool
    ) -> None:
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.valid_area = valid_area
        self.contiguous = contiguous

    def _forward(self, inputs: tuple[Tensor, ...]) -> Tensor:
        self.src = inputs[0]
        return Tensor(
            self._forward_code(),
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            valid_area=self.valid_area,
            contiguous=self.contiguous,
            inputs=(self.src,),
            backprop_ctx=self,
            dtype=self.src.dtype,
        )

    def _forward_code(self) -> TensorOpCode:
        raise NotImplementedError()

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        raise NotImplementedError()


class Permute(DifferentiableView):
    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        valid_area: tuple[tuple[int, int], ...] | None,
        contiguous: bool,
        axes: tuple[int, ...],
    ) -> None:
        self.axes = axes
        super().__init__(shape, strides, offset, valid_area, contiguous)

    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.PERMUTE

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        # argsort https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
        perm = sorted(range(len(self.axes)), key=self.axes.__getitem__)
        return (grad.permute(*perm),)


class Expand(DifferentiableView):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.EXPAND

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        orig_shape = self.src.shape
        added_axis = tuple(range(grad.ndim - len(orig_shape)))
        expanded_axis = tuple([i + len(added_axis) for i, s in enumerate(orig_shape) if s == 1])
        y = grad.sum(axis=added_axis + expanded_axis, keepdims=True)
        return y if not added_axis else y.reshape(*[s for i, s in enumerate(y.shape) if i not in added_axis])


class Crop(DifferentiableView):
    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        valid_area: tuple[tuple[int, int], ...] | None,
        contiguous: bool,
        crop_area: tuple[tuple[int, int], ...],
    ) -> None:
        self.crop_area = crop_area
        super().__init__(shape, strides, offset, valid_area, contiguous)

    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.CROP

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return grad.pad(tuple([(c[0], s - c[1]) for s, c in zip(self.src.shape, self.crop_area)]))


class Pad(DifferentiableView):
    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        valid_area: tuple[tuple[int, int], ...] | None,
        contiguous: bool,
        padding: tuple[tuple[int, int], ...],
    ) -> None:
        self.padding = padding
        super().__init__(shape, strides, offset, valid_area, contiguous)

    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.PAD

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad.crop(tuple([(p[0], s + p[0]) for s, p in zip(self.src.shape, self.padding)])),)


class Reshape(DifferentiableView):
    def _forward_code(self) -> TensorOpCode:
        return TensorOpCode.RESHAPE

    def _backward(self, grad: Tensor) -> tuple[Tensor, ...]:
        return (grad.reshape(*self.src.shape),)


class Tensor:
    #
    # constructors
    #

    def __init__(
        self,
        arg: int | float | list | bool | TensorOpCode,
        shape: tuple[int, ...] | None = None,
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
        valid_area: tuple[tuple[int, int], ...] | None = None,
        contiguous: bool = True,
        inputs: tuple[Tensor, ...] | None = None,
        backprop_ctx: Differentiable | None = None,
        dtype: DType = dtypes.float32,
    ):
        self.dev = get_device()
        self.grad: Tensor | None = None

        if isinstance(arg, TensorOpCode):
            self.code: TensorOpCode = arg
            self.shape: tuple[int, ...] = shape if shape is not None else ()
            self.strides: tuple[int, ...] = strides if strides is not None else shape_to_strides(self.shape)
            self.offset = offset
            self.valid_area = tuple([(0, s) for s in self.shape]) if valid_area is None else valid_area
            self.contiguous = contiguous
            self.inputs = inputs if inputs is not None else ()
            self.backprop_ctx: Differentiable | None = backprop_ctx
            self.dtype = dtype

            self.cpu_buffer: device.CPUMemoryBuffer | None = None
            self.dev_buffer: device.DeviceMemoryBuffer | None = None
            self.materialized: bool = False
            return

        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, list) or isinstance(arg, bool):
            self.init_from_data(arg, shape=shape)
            return

        raise TypeError(f"cannot handle type {type(arg)} in Tensor constructor")

    def init_from_data(self, data: int | float | list | bool, shape=None):
        self.code = TensorOpCode.BUFFER
        self.offset = 0
        self.contiguous = True
        self.inputs = ()
        self.dev_buffer = None
        self.backprop_ctx = None
        self.materialized = True

        # scalar

        if isinstance(data, bool):
            if shape:
                raise RuntimeError(f"shape {shape} must not be passed to scalar initialization")
            self.shape = ()
            self.strides = ()
            self.valid_area = ()
            self.cpu_buffer = device.CPUMemoryBuffer([1.0 if data else 0.0])
            self.dtype = dtypes.bool
            return

        if isinstance(data, float) or isinstance(data, int):
            if shape:
                raise RuntimeError(f"shape {shape} must not be passed to scalar initialization")
            self.shape = ()
            self.strides = ()
            self.valid_area = ()
            self.cpu_buffer = device.CPUMemoryBuffer([data] if isinstance(data, float) else [float(data)])
            self.dtype = dtypes.float32
            return

        # tensor
        if isinstance(data, list):
            flattened: list[typing.Any] = []
            actual_shape: list[int] = []

            def f(d, dim):
                if not isinstance(d, int) and not isinstance(d, float) and not isinstance(d, bool) and not isinstance(d, list):
                    raise ValueError(f"array must be a multi-dimensional array of int or float or bool: {data}")

                if isinstance(d, bool):
                    flattened.append(d)
                    return

                if isinstance(d, int) or isinstance(d, float):
                    flattened.append(d if isinstance(d, float) else float(d))
                    return

                # d must be list here
                length = len(d)
                if len(actual_shape) == dim:
                    actual_shape.append(length)
                elif length != actual_shape[dim]:
                    raise ValueError(f"array must be homogeneous: {data}")

                for elem in d:
                    f(elem, dim + 1)

            f(data, 0)

            self.dtype = dtypes.bool if isinstance(flattened[0], bool) else dtypes.float32
            if self.dtype == dtypes.bool:
                flattened = [1.0 if v else 0.0 for v in flattened]
            self.shape = shape if shape is not None else tuple(actual_shape)
            self.strides = shape_to_strides(self.shape)
            self.valid_area = tuple([(0, s) for s in self.shape])
            self.cpu_buffer = device.CPUMemoryBuffer(flattened)
            return

        raise TypeError(f"type {type(data)} is unsupported as Tensor")

    @classmethod
    def new_buffer_op(cls, data: typing.Any, shape: tuple[int, ...] | None = None) -> Tensor:
        return Tensor(data, shape=shape)

    @classmethod
    def new_data_op(cls, d: DifferentiableData, src: Tensor) -> Tensor:
        return d.forward((src,))

    @classmethod
    def new_binary_op(cls, d: DifferentiableBinary, l: Tensor, r: Tensor) -> Tensor:
        return d.forward((l, r))

    @classmethod
    def new_unary_op(cls, d: DifferentiableUnary, src: Tensor) -> Tensor:
        return d.forward((src,))

    @classmethod
    def new_reduce_op(cls, d: DifferentiableReduce, src: Tensor) -> Tensor:
        return d.forward((src,))

    @classmethod
    def new_view_op(cls, d: DifferentiableView, src: Tensor) -> Tensor:
        return d.forward((src,))

    @classmethod
    def full(cls, shape: tuple[int, ...], val: float):
        return Tensor.new_buffer_op([val] * math.prod(shape), shape=shape)

    @classmethod
    def full_like(cls, t: Tensor, val: float):
        return Tensor.full(t.shape, val)

    @classmethod
    def ones_like(cls, t: Tensor):
        return Tensor.full_like(t, 1.0)

    @classmethod
    def arange(cls, n: int):
        return Tensor([i for i in range(n)])

    @classmethod
    def wrap(cls, x: typing.Any):
        return x if isinstance(x, Tensor) else Tensor(x)

    #
    # properties
    #

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    #
    # arith and math
    #

    def add(self, r: Tensor):
        r = Tensor.wrap(r)
        l, r = self.broadcasted(r)
        return Tensor.new_binary_op(Add(), l, r)

    def sub(self, r: Tensor):
        return self + (-Tensor.wrap(r))

    def mul(self, r: Tensor):
        r = Tensor.wrap(r)
        l, r = self.broadcasted(r)
        return Tensor.new_binary_op(Mul(), l, r)

    def truediv(self, r: Tensor):
        # l/r = l * (1/r)
        return self * Tensor.wrap(r).recip()

    def recip(self):
        return Tensor.new_unary_op(Recip(), self)

    def pow(self, r: Tensor):
        r = Tensor.wrap(r)
        l, r = self.broadcasted(r)
        return Tensor.new_binary_op(Pow(), l, r)

    def matmul(self, r: Tensor):
        r = Tensor.wrap(r)
        if self.ndim != 2 or r.ndim != 2:
            raise RuntimeError(f"matmul arg must be 2D matrix, got {self.shape} and {r.shape}")

        if self.shape[1] != r.shape[0]:
            raise RuntimeError(f"invalid shape combination for matmul, got {self.shape} and {r.shape}")

        # using matmul trick, see https://mesozoic-egg.github.io/tinygrad-notes/20241203_matmul.html

        _l = self.reshape(self.shape[0], 1, self.shape[1]).broadcast_to((self.shape[0], r.shape[1], self.shape[1]))
        _r = r.reshape(1, r.shape[0], r.shape[1]).transpose(1, 2).broadcast_to((_l.shape[0], r.shape[1], r.shape[0]))
        return (_l * _r).sum(axis=2)

    def maximum(self, r: Tensor):
        l, r = self.broadcasted(Tensor.wrap(r))
        return Tensor.new_binary_op(Maximum(), l, r)

    def minimum(self, r: Tensor):
        return -((-self).maximum(-r))

    def neg(self):
        return self * -1

    def eq(self, r: Tensor) -> Tensor:
        return self.ne(Tensor.wrap(r)).logical_not()

    def ne(self, r: Tensor) -> Tensor:
        l, r = self.broadcasted(Tensor.wrap(r))
        return Tensor.new_binary_op(Ne(), l, r)

    def logical_not(self) -> Tensor:
        return self.ne(Tensor.wrap(True))

    def log(self):
        return Tensor.new_unary_op(Log(), self)

    def sin(self):
        return Tensor.new_unary_op(Sin(), self)

    def cos(self):
        return Tensor.new_unary_op(Cos(), self)

    def tanh(self):
        return Tensor.new_unary_op(Tanh(), self)

    def exp(self):
        return Tensor.new_unary_op(Exp(), self)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        return self._reduce(Sum, axis, keepdims)

    def prod(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        return self._reduce(Prod, axis, keepdims)

    def max(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        return self._reduce(Max, axis, keepdims)

    def min(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
        return -(-self.max(axis=axis, keepdims=keepdims))

    def _reduce(self, red: typing.Type[DifferentiableReduce], axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        # this parallel reduction should be optimized
        if axis is None:
            axis = tuple(list(range(self.ndim)))

        if isinstance(axis, tuple):
            if self.ndim < len(axis):
                raise RuntimeError(f"too many axis {axis} for shape {self.shape}")

            if len(axis) == 0:
                return self

            t = self
            for a in sorted(list(axis), reverse=True):
                t = t._reduce(red, axis=a, keepdims=keepdims)

            return t

        if axis < 0 or self.ndim - 1 < axis:
            raise RuntimeError(f"invalid axis {axis} for shape {self.shape}")

        t = Tensor.new_reduce_op(red(axis, keepdims), self)
        if keepdims:
            return t

        newshape = list(t.shape)
        newshape = newshape[:axis] + newshape[axis + 1 :]
        return t.reshape(*newshape)

    #
    # shape movement
    #

    def permute(self, *axes: int) -> Tensor:
        if sorted(axes) != list(range(self.ndim)):
            raise RuntimeError(f"permute axes must be wrong: {axes}")

        newshape = []
        newstrides = []
        newvalidarea = []
        for a in axes:
            newshape.append(self.shape[a])
            newstrides.append(self.strides[a])
            newvalidarea.append(self.valid_area[a])

        return Tensor.new_view_op(Permute(tuple(newshape), tuple(newstrides), self.offset, tuple(newvalidarea), False, axes), self)

    def transpose(self, axis0: int = 1, axis1: int = 0) -> Tensor:
        if axis0 < 0 or axis1 < 0 or self.ndim <= axis0 or self.ndim <= axis1:
            raise RuntimeError(f"transpose axes out of bounds: {axis0} and {axis1}")

        axes = list(range(self.ndim))
        axes[axis0], axes[axis1] = axes[axis1], axes[axis0]
        return self.permute(*axes)

    def _broadcasted_shape(self, s2: tuple[int, ...]) -> tuple[int, ...]:
        ls1 = list(self.shape[:])
        ls2 = list(s2)
        maxlen = max(len(ls1), len(ls2))
        ls1 = [1] * (maxlen - len(ls1)) + ls1
        ls2 = [1] * (maxlen - len(ls2)) + ls2

        shape = []
        for d1, d2 in zip(ls1, ls2):
            if d1 == d2:
                shape.append(d1)
                continue

            if d1 == 1:
                shape.append(d2)
                continue

            if d2 == 1:
                shape.append(d1)
                continue

            raise RuntimeError(f"shapes are not broadcastable: {self.shape} and {s2}")

        return tuple(shape)

    def broadcast_to(self, shape: tuple[int, ...]) -> Tensor:
        if len(self.shape) > len(shape):
            raise RuntimeError(f"broadcasting {self.shape} to {shape} is impossible")

        delta = len(shape) - len(self.shape)
        expanded_shape = [1] * delta + list(self.shape[:])
        expanded_strides = [0] * delta + list(self.strides[:])
        expanded_valid_area = [(0, 1)] * delta + list(self.valid_area[:])

        newstrides = tuple([st if shape[i] == expanded_shape[i] else 0 for i, st in enumerate(expanded_strides)])
        newvalidarea = tuple([va if shape[i] == expanded_shape[i] else (0, shape[i]) for i, va in enumerate(expanded_valid_area)])
        return Tensor.new_view_op(Expand(shape, newstrides, self.offset, newvalidarea, False), self)

    def broadcasted(self, t: Tensor) -> tuple[Tensor, Tensor]:
        newshape = self._broadcasted_shape(t.shape)
        l, r = self, t
        if l.shape != newshape:
            l = l.broadcast_to(newshape)
        if r.shape != newshape:
            r = r.broadcast_to(newshape)
        return l, r

    def crop(self, areas: tuple[tuple[int, int] | None, ...]):
        if len(areas) != self.ndim:
            raise RuntimeError("crop area size must be the same with ndim")

        arg = tuple([(0, s) if area is None else (area[0], area[1]) for s, area in zip(self.shape, areas)])
        newshape = []
        newoffset = self.offset
        for i, a in enumerate(arg):
            if a[1] <= a[0] or a[0] < 0 or self.shape[i] < a[1]:
                raise RuntimeError(f"invalid crop arg {a} for axis {i}, dim is {self.shape[i]}")
            newshape.append(max(0, (a[1] - a[0])))
            newoffset += a[0] * self.strides[i]

        return Tensor.new_view_op(Crop(tuple(newshape), self.strides, newoffset, None, False, arg), self)

    def pad(self, padding: tuple[tuple[int, int], ...]):
        if len(padding) != self.ndim:
            raise RuntimeError("padding size must be the same with ndim")

        arg = tuple([(0, 0) if p is None else p for p in padding])
        newshape = []
        newoffset = self.offset
        newvalidarea = []
        for pd, sp, st, va in zip(arg, self.shape, self.strides, self.valid_area):
            if len(pd) != 2:
                raise RuntimeError(f"padding must be (start, stop), got {pd}")

            if pd[0] < 0 or pd[1] < 0:
                raise RuntimeError("padding must not be negative")
            newshape.append(sp + pd[0] + pd[1])
            newoffset -= pd[0] * st
            newvalidarea.append((va[0] + pd[0], va[1] + pd[0]))

        return Tensor.new_view_op(Pad(tuple(newshape), self.strides, newoffset, tuple(newvalidarea), False, arg), self)

    def reshape(self, *shape: int) -> Tensor:
        if math.prod(shape) != self.size:
            raise RuntimeError(f"invalid reshape {shape} for size {self.size} tensor")

        return Tensor.new_view_op(Reshape(tuple(shape), shape_to_strides(shape), 0, None, True), self if self.contiguous else self.copy())

    def _getitem(self, indices):
        if type(indices) != tuple:
            indices = (indices,)

        if self.ndim < len(indices):
            raise ValueError(f"too many index accessors specified")

        if Tensor in [type(i) for i in indices]:
            raise NotImplementedError("advanced index is still not implemented")

        return self._get_item_basic(indices)

    # numpy basic index
    # https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
    def _get_item_basic(self, indices):
        for i in range(self.ndim - len(indices)):
            indices = indices + (slice(None, None, None),)

        t = self
        curdim = 0
        for idx in indices:
            if type(idx) == int:
                if t.shape[curdim] - 1 < idx:
                    raise ValueError(f"index out of bounds for axis {curdim} with size {t.shape[curdim]}")
                idx = idx if 0 <= idx else t.shape[curdim] + idx
                start, stop = idx, idx + 1
            elif type(idx) == slice:
                start = 0 if idx.start is None else t.shape[curdim] + idx.start if idx.start < 0 else idx.start
                stop = t.shape[curdim] if idx.stop is None else t.shape[curdim] + idx.stop if idx.stop < 0 else idx.stop
                stop = min(t.shape[curdim], stop)
                # todo: support step
            else:
                raise RuntimeError(f"unhandled index in basic index: {type(idx)}")

            t = t.crop(tuple([(start, stop) if i == curdim else None for i in range(t.ndim)]))

            if type(idx) == int:
                t = t.reshape(*t.shape[:curdim], *t.shape[curdim + 1 :])
            else:
                curdim += 1

        return t

    #
    # data operation
    #

    def copy(self):
        return Tensor.new_data_op(Copy(), self)

    def to(self, dt: DType) -> Tensor:
        if self.dtype == dt:
            return self
        if self.dtype == dtypes.bool:
            cp = self.copy()
            cp.dtype = dt
            return cp
        if self.dtype == dtypes.float32:
            cp = self.copy()
            cp.dtype = dt
            return cp

        raise RuntimeError(f"cast from {self.dtype} to {dt} still not implemented")

    #
    # operators
    #

    def __add__(self, r):
        return self.add(Tensor.wrap(r))

    def __radd__(self, l):
        return Tensor.wrap(l).add(self)

    def __sub__(self, r):
        return self.sub(Tensor.wrap(r))

    def __rsub__(self, l):
        return Tensor.wrap(l).sub(self)

    def __mul__(self, r):
        return self.mul(r)

    def __rmul__(self, l):
        return Tensor.wrap(l).mul(self)

    def __truediv__(self, r):
        return self.truediv(r)

    def __rtruediv__(self, l):
        return Tensor.wrap(l).truediv(self)

    def __pow__(self, r):
        return self.pow(r)

    def __rpow__(self, l):
        return Tensor.wrap(l).pow(self)

    def __matmul__(self, r):
        return self.matmul(r)

    def __neg__(self):
        return self.neg()

    def __ne__(self, r):
        return self.ne(r)

    def __lt__(self, r):
        l, r = self.broadcasted(Tensor.wrap(r))
        return Tensor.new_binary_op(Lt(), l, r)

    def __gt__(self, r):
        return Tensor.wrap(r).__lt__(self)

    def __le__(self, r):
        return (self > r).logical_not()

    def __ge__(self, r):
        return (self < r).logical_not()

    def __getitem__(self, indices):
        return self._getitem(indices)

    def __str__(self):
        def f(depth: int, t: Tensor):
            indent = "    " * depth
            trail_comma = "," if depth != 0 else ""

            if not t.inputs:
                return f"{indent}{t.str_oneline()}{trail_comma}"

            inputs = "[\n" + "\n".join([f(depth + 1, i) for i in t.inputs]) + f"\n{indent}]"
            return f"{indent}{t.str_oneline().rstrip(')')} inputs: {inputs}{trail_comma}"

        return f(0, self)

    def str_oneline(self):
        return f"Tensor(code: {self.code} {self.shape}_{self.strides}_{self.offset}_{self.valid_area} cpu_buff:{self.cpu_buffer.raw if self.cpu_buffer else 'x'} dev_buff:{'o' if self.dev_buffer else 'x'})"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        if hasattr(self, "dev_buffer") and self.dev_buffer:
            self.dev.free(self.dev_buffer)

    #
    # materialization
    #

    def materialize(self):
        if not self.materialized:
            Materializer.materialize(self)
            self.materialized = True
        return self

    def tolist(self):
        if not self.materialized:
            self.materialize()

        if self.cpu_buffer is None:
            self.cpu_buffer = device.CPUMemoryBuffer(None)
            self.dev.copy_from_device(self.dev_buffer, self.cpu_buffer)

        # calculate index combination by shape
        indices = [list(c) for c in list(itertools.product(*[range(n) for n in self.shape]))]

        # pick the actual value to be in the result by the index
        vals = []
        for idx in indices:
            if all([area_dim[0] <= i_dim and i_dim < area_dim[1] for i_dim, area_dim in zip(idx, self.valid_area)]):
                assert self.cpu_buffer.raw is not None
                val = self.cpu_buffer.raw[self.offset + sum([st * i for st, i in zip(self.strides, idx)])]
                vals.append(val if self.dtype == dtypes.float32 else bool(val))
            else:
                # if idx is not in valid area, use invalid value
                vals.append(0.0 if self.dtype == dtypes.float32 else False)

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
            self.grad = Tensor.ones_like(self)

        differentiables = []
        seen = set()

        def dfs(_t: Tensor):
            if _t.backprop_ctx is None or _t in seen:
                return

            seen.add(_t)
            for i in _t.inputs:
                dfs(i)
            differentiables.append(_t.backprop_ctx)

        dfs(self)
        differentiables.reverse()

        for d in differentiables:
            gy = d.output.grad
            assert gy is not None
            gxs = d.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(d.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx

    def backward(self, gradient=None):
        self.backprop()


class Instruction:
    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != "instid")
        return f"<{self.__class__.__name__}({params})>"


@dataclass
class AllocateDeviceMemory(Instruction):
    t: Tensor


@dataclass
class CopyBufferPythonToDevice(Instruction):
    t: Tensor


@dataclass
class CopyBufferDeviceToPython(Instruction):
    t: Tensor


@dataclass
class CopyDevicePointer(Instruction):
    src: Tensor
    dst: Tensor


@dataclass
class InvokeUnaryKernel(Instruction):
    kern_name: str
    dst: Tensor
    src: Tensor


@dataclass
class InvokeBinaryKernel(Instruction):
    kern_name: str
    dst: Tensor
    srcl: Tensor
    srcr: Tensor


@dataclass
class InvokeReduceKernel(Instruction):
    kern_name: str
    dst: Tensor
    src: Tensor
    axis: int


class Materializer:
    # materializer is singleton for caching some info
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return

        if backend == Backend.CUDA:
            self.kernel_generator = cuda.CodeGenerator()
            self.kernel_manager = cuda.KernelManager()

        self.initialized = True

    @classmethod
    def materialize(cls, t: Tensor) -> None:
        self = cls()  # get materializer singleton instance

        if dbg:
            print("=== materialization start ===")
            print("=== backend")
            print(backend)

            print("=== tensor graph")
            print(t)

        tensors = self.linearize(t)

        if dbg:
            print("=== linearized tensors")
            print("\n".join([f"{t.str_oneline()}" for t in tensors]))

        # generate and load kernels
        kerns = self.generate_kernels(tensors)

        if dbg:
            print("=== kernels")
            print("\n\n".join([f"{kern.src}" for kern in kerns]))

        self.kernel_manager.load(kerns)

        insts = self.generate_instructions(tensors)
        if dbg:
            print("=== instructions")
            print("\n".join([f"{inst}" for inst in insts]))

        if dbg:
            print("=== execution logs")
        self.execute(insts)

        if dbg:
            print("=== materialized tensor")
            print(t.str_oneline())
            print("=== materialization successfully finished ===")

    def linearize(self, t: Tensor):
        tensors = []
        seen = set()

        def dfs(_t: Tensor):
            if _t in seen:
                return

            seen.add(_t)

            if _t.inputs is not None:
                for i in _t.inputs:
                    dfs(i)

            tensors.append(_t)

        dfs(t)
        return tensors

    def to_kern_opcode(self, c: TensorOpCode) -> kernel.OpCode:
        d = {
            TensorOpCode.RECIP: kernel.OpCode.RECIP,
            TensorOpCode.ADD: kernel.OpCode.ADD,
            TensorOpCode.MUL: kernel.OpCode.MUL,
            TensorOpCode.POW: kernel.OpCode.POW,
            TensorOpCode.LOG: kernel.OpCode.LOG,
            TensorOpCode.COPY: kernel.OpCode.COPY,
            TensorOpCode.SUM: kernel.OpCode.SUM,
            TensorOpCode.PROD: kernel.OpCode.PROD,
            TensorOpCode.MAX: kernel.OpCode.MAX,
            TensorOpCode.SIN: kernel.OpCode.SIN,
            TensorOpCode.COS: kernel.OpCode.COS,
            TensorOpCode.TANH: kernel.OpCode.TANH,
            TensorOpCode.EXP: kernel.OpCode.EXP,
            TensorOpCode.NE: kernel.OpCode.NE,
            TensorOpCode.LT: kernel.OpCode.LT,
            TensorOpCode.MAXIMUM: kernel.OpCode.MAXIMUM,
        }
        return d[c]

    def generate_kernels(self, tensors: list[Tensor]) -> list[kernel.Kernel]:
        kerns: list[kernel.Kernel] = []
        for t in tensors:
            if t.code.is_unary_op():
                name, src = self.kernel_generator.generate_unary_kernel(self.to_kern_opcode(t.code), t.ndim)

            elif t.code.is_binary_op():
                name, src = self.kernel_generator.generate_binary_kernel(self.to_kern_opcode(t.code), t.ndim)

            elif t.code.is_reduce_op():
                assert t.backprop_ctx is not None
                assert hasattr(t.backprop_ctx, "axis")
                name, src = self.kernel_generator.generate_reduce_kernel(self.to_kern_opcode(t.code), t.ndim, t.backprop_ctx.axis)

            else:
                continue

            if name not in [k.name for k in kerns]:
                kerns.append(kernel.Kernel(name, src, None))

        return kerns

    def generate_instructions(self, tensors: list[Tensor]) -> list[Instruction]:
        insts: list[Instruction] = []
        for t in tensors:
            if t.code.is_buffer_op():
                insts.append(AllocateDeviceMemory(t))
                insts.append(CopyBufferPythonToDevice(t))

            elif t.code.is_unary_op():
                insts.append(AllocateDeviceMemory(t))
                insts.append(InvokeUnaryKernel(kernel.to_kern_name(self.to_kern_opcode(t.code), t.ndim), t, t.inputs[0]))

            elif t.code.is_binary_op():
                insts.append(AllocateDeviceMemory(t))
                insts.append(InvokeBinaryKernel(kernel.to_kern_name(self.to_kern_opcode(t.code), t.ndim), t, t.inputs[0], t.inputs[1]))

            elif t.code.is_reduce_op():
                insts.append(AllocateDeviceMemory(t))
                assert t.backprop_ctx is not None
                assert hasattr(t.backprop_ctx, "axis")
                axis = t.backprop_ctx.axis
                insts.append(InvokeReduceKernel(kernel.to_reduce_kern_name(self.to_kern_opcode(t.code), t.ndim, axis), t, t.inputs[0], axis))

            elif t.code.is_view_op():
                insts.append(CopyDevicePointer(t.inputs[0], t))

        return insts

    def execute(self, insts: list[Instruction]) -> None:
        def grid_block(total: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
            threads_per_block = 256
            total_blocks = (total + threads_per_block - 1) // threads_per_block
            grid_x = min(total_blocks, 65535)
            grid_y = (total_blocks + grid_x - 1) // grid_x
            return (grid_x, grid_y, 1), (threads_per_block, 1, 1)

        for inst in insts:
            match inst:
                case AllocateDeviceMemory():
                    inst.t.dev_buffer = inst.t.dev.allocate(inst.t.size)

                case CopyBufferPythonToDevice():
                    inst.t.dev.copy_to_device(inst.t.cpu_buffer, inst.t.dev_buffer)

                case CopyBufferDeviceToPython():
                    inst.t.dev.copy_to_device(inst.t.dev_buffer, inst.t.cpu_buffer)

                case CopyDevicePointer():
                    inst.dst.dev_buffer = inst.src.dev_buffer

                case InvokeUnaryKernel():
                    params = (
                        inst.dst.size,
                        *inst.dst.shape,
                        *inst.src.strides,
                        *sum(inst.src.valid_area, ()),
                        inst.src.offset,
                        inst.src.dev_buffer,
                        inst.dst.dev_buffer,
                    )
                    grid, block = grid_block(inst.dst.size)
                    if dbg:
                        print(f"invoking kernel: {inst.kern_name=}, {grid=}, {block=}, {params=}")
                    self.kernel_manager.invoke(inst.kern_name, grid, block, params)

                case InvokeBinaryKernel():
                    params = (
                        inst.dst.size,
                        *inst.dst.shape,
                        *inst.srcl.strides,
                        *inst.srcr.strides,
                        *sum(inst.srcl.valid_area, ()),
                        *sum(inst.srcr.valid_area, ()),
                        inst.srcl.offset,
                        inst.srcr.offset,
                        inst.srcl.dev_buffer,
                        inst.srcr.dev_buffer,
                        inst.dst.dev_buffer,
                    )
                    grid, block = grid_block(inst.dst.size)
                    if dbg:
                        print(f"invoking kernel: {inst.kern_name=}, {grid=}, {block=}, {params=}")
                    self.kernel_manager.invoke(inst.kern_name, grid, block, params)

                case InvokeReduceKernel():
                    params = (
                        inst.dst.size,
                        inst.src.shape[inst.axis],
                        *inst.dst.shape,
                        *inst.src.strides,
                        *sum(inst.src.valid_area, ()),
                        inst.src.offset,
                        inst.src.dev_buffer,
                        inst.dst.dev_buffer,
                    )
                    grid, block = grid_block(inst.dst.size)
                    if dbg:
                        print(f"invoking kernel: {inst.kern_name=}, {grid=}, {block=}, {params=}")
                    self.kernel_manager.invoke(inst.kern_name, grid, block, params)
            if dbg:
                print(f"executed: {inst}")


def tensor(arr, requires_grad=False):
    return Tensor(arr)


def ones_like(t: Tensor):
    return Tensor.ones_like(t)


if __name__ == "__main__":
    pass
