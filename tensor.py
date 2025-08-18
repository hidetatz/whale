from __future__ import annotations

import collections
import itertools
import math
import os
import time
from dataclasses import dataclass
from enum import IntEnum, auto

import cuda
import device
import kernel
from backend import Backend

dbg = os.getenv("WHALE_DEBUG", "") != ""

backend = Backend.detect()

cuda_device = cuda.Device()


def get_device():
    if backend == Backend.CUDA:
        return cuda_device

    raise RuntimeError("no backend")


def shape_to_strides(shape: list[int]):
    return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])


class TensorOpCode(IntEnum):
    _buffer_op_start = auto()
    BUFFER = auto()
    _buffer_op_end = auto()

    _unary_op_start = auto()
    RECIP = auto()
    LOG = auto()
    _unary_op_end = auto()

    _binary_op_start = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()
    _binary_op_end = auto()

    _view_op_start = auto()
    VIEW_AS = auto()
    _view_op_end = auto()

    def _in(self, start, end):
        return start < self.value and self.value < end

    def is_buffer_op(self):
        return self._in(TensorOpCode._buffer_op_start, TensorOpCode._buffer_op_end)

    def is_unary_op(self):
        return self._in(TensorOpCode._unary_op_start, TensorOpCode._unary_op_end)

    def is_binary_op(self):
        return self._in(TensorOpCode._binary_op_start, TensorOpCode._binary_op_end)

    def is_view_op(self):
        return self._in(TensorOpCode._view_op_start, TensorOpCode._view_op_end)

    def __str__(self):
        return self.name


class Differentiable:
    def forward(self, inputs: tuple(Tensor)):
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
class DifferentiableUnary(Differentiable):
    def _forward(self, src: Tensor) -> Tensor:
        self.src = src
        return Tensor(self._forward_code(), shape=src.shape, inputs=(src,), backprop_ctx=self, generation=src.generation)

    def _backward(self, grad: Tensor) -> tuple(Tensor):
        raise NotImplementedError()


class Recip(DifferentiableUnary):
    def _forward_code(self):
        return TensorOpCode.RECIP

    def _backward(self, grad):
        return grad * (Tensor.full_like(self.src, -1.0) / (self.src * self.src))


class Log(DifferentiableUnary):
    def _forward_code(self):
        return TensorOpCode.LOG

    def _backward(self, grad):
        return grad / self.src


# binary
class DifferentiableBinary(Differentiable):
    def _forward(self, l: Tensor, r: Tensor) -> Tensor:
        self.l = l
        self.r = r
        return Tensor(self._forward_code(), shape=l.shape, inputs=(l, r), backprop_ctx=self, generation=max(l.generation, r.generation))

    def _backward(self, grad: Tensor) -> tuple(Tensor):
        raise NotImplementedError()


class Add(DifferentiableBinary):
    def _forward_code(self):
        return TensorOpCode.ADD

    def _backward(self, grad: Tensor):
        return grad, grad


class Mul(DifferentiableBinary):
    def _forward_code(self):
        return TensorOpCode.MUL

    def _backward(self, grad: Tensor):
        return grad * self.r, grad * self.l


class Pow(DifferentiableBinary):
    def _forward_code(self):
        return TensorOpCode.POW

    def _backward(self, grad: Tensor) -> Tensor:
        lgrad = grad * self.r * (self.l ** (self.r - Tensor.full_like(self.r, 1)))
        rgrad = grad * (self.l**self.r) * self.l.log()
        return lgrad, rgrad


# view
class DifferentiableView(Differentiable):
    def _backward(self, grad: Tensor) -> tuple(Tensor):
        raise NotImplementedError()


class ViewAs(Differentiable):
    def __init__(self, shape: tuple[int], strides: tuple[int], offset: int):
        self.shape = shape
        self.strides = strides
        self.offset = offset
        super().__init__()

    def _forward(self, src: Tensor) -> Tensor:
        self.src = src
        return Tensor(
            TensorOpCode.VIEW_AS,
            shape=self.shape,
            strides=self.strides,
            offset=self.offset,
            inputs=(src,),
            backprop_ctx=self,
            generation=src.generation,
        )

    def _backward(self, grad: Tensor) -> tuple(Tensor):
        raise NotImplementedError()


class Tensor:
    #
    # constructors
    #

    def __init__(
        self,
        arg: int | float | list | TensorOpcode,
        shape: tuple[int] = None,
        strides: tuple[int] = None,
        offset: int = 0,
        inputs: tuple[Tensor] = [],
        backprop_ctx=None,
        generation: int = 0,
    ):
        self.dev = get_device()
        self.grad: Tensor = None

        if isinstance(arg, TensorOpCode):
            self.code: TensorOpCode = arg
            self.shape: tuple[int] = shape
            self.strides: tuple[int] = strides if strides is not None else shape_to_strides(shape)
            self.offset: int = offset
            self.inputs: list[Tensor] = inputs
            self.backprop_ctx: Differentiable = backprop_ctx
            self.generation: int = generation

            self.cpu_buffer: device.CPUMemoryBuffer = None
            self.dev_buffer: device.DeviceMemoryBuffer = None
            self.materialized: bool = False
            return

        if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, list):
            self.init_from_data(arg, shape=shape)
            return

        raise TypeError(f"cannot handle type {type(arg)} in Tensor constructor")

    def init_from_data(self, data: int | float | list, shape=None):
        self.code = TensorOpCode.BUFFER
        self.offset = 0
        self.inputs = []
        self.dev_buffer = None
        self.backprop_ctx = None
        self.generation = 0
        self.materialized = True

        # scalar
        if isinstance(data, float) | isinstance(data, int):
            if shape:
                raise RuntimeError(f"shape {shape} must not be passed to scalar initialization")
            self.shape = ()
            self.strides = ()
            self.cpu_buffer = device.CPUMemoryBuffer([data] if isinstance(data, float) else [float(data)])
            return

        # tensor
        if isinstance(data, list):
            flattened = []
            actual_shape = []

            def f(d, dim):
                if not isinstance(d, int) and not isinstance(d, float) and not isinstance(d, list):
                    raise ValueError(f"array must be a multi-dimensional array of int or float: {data}")

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

            self.shape = shape if shape is not None else tuple(actual_shape)
            self.strides = shape_to_strides(self.shape)
            self.cpu_buffer = device.CPUMemoryBuffer(flattened)
            return

        raise TypeError(f"type {type(data)} is unsupported as Tensor")

    @classmethod
    def new_buffer_op(cls, data: any, shape: tuple(int) = None) -> Tensor:
        return Tensor(data, shape=shape)

    @classmethod
    def new_binary_op(cls, d: DifferentiableBinary, l: Tensor, r: Tensor) -> Tensor:
        return d.forward((l, r))

    @classmethod
    def new_unary_op(cls, d: DifferentiableUnary, src: Tensor) -> Tensor:
        return d.forward((src,))

    @classmethod
    def new_view_op(cls, d: DifferentiableView, src: Tensor) -> Tensor:
        return d.forward((src))

    @classmethod
    def full(cls, shape: tuple[int], val: float):
        return Tensor.new_buffer_op([val] * math.prod(shape), shape=shape)

    @classmethod
    def full_like(cls, t: Tensor, val: float):
        return Tensor.full(t.shape, val)

    @classmethod
    def ones_like(cls, t: Tensor):
        return Tensor.full_like(t, 1.0)

    #
    # properties
    #

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def _is_scalar(self):
        return self.ndim == 0

    #
    # operators
    #

    def add(self, r: Tensor):
        return Tensor.new_binary_op(Add(), self, r)

    def sub(self, r: Tensor):
        return self + (-r)

    def mul(self, r: Tensor):
        return Tensor.new_binary_op(Mul(), self, r)

    def truediv(self, r: Tensor):
        # l/r = l * (1/r)
        return self * r.recip()

    def recip(self):
        return Tensor.new_unary_op(Recip(), self)

    def log(self):
        return Tensor.new_unary_op(Log(), self)

    def pow(self, r: Tensor):
        return Tensor.new_binary_op(Pow(), self, r)

    def neg(self):
        return self * Tensor.full_like(self, -1)

    def _getitem(self, indices):
        if type(indices) != tuple:
            indices = (indices,)

        if self._is_scalar():
            raise ValueError(f"cannot index on scalar: {self}")

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
        return Tensor.new_view_op(ViewAs(newshape, newstrides, newoffset), [self])

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

    def __neg__(self):
        return self.neg()

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
        return f"Tensor(code: {self.code} {self.shape}_{self.strides}_{self.offset} cpu_buff:{self.cpu_buffer.raw if self.cpu_buffer else 'x'} dev_buff:{'o' if self.dev_buffer else 'x'})"

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
            vals.append(self.cpu_buffer.raw[self.offset + sum([st * i for st, i in zip(self.strides, idx)])])

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

        def f(d: Differentiable):
            if d not in seen:
                differentiables.append(d)
                seen.add(d)
                differentiables.sort(key=lambda x: x.generation)

        f(self.backprop_ctx)

        while differentiables:
            d = differentiables.pop()
            gy = d.output.grad
            gxs = d.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(d.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.backprop_ctx is not None:
                    f(x.backprop_ctx)

    def backward(self):
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

            for i in _t.inputs:
                dfs(i)

            tensors.append(_t)

        dfs(t)
        return tensors

    def generate_kernels(self, tensors: list[Tensor]) -> list[kernel.Kernel]:
        code_map: dict[TensorOpCode, kernel.OpCode] = {
            TensorOpCode.RECIP: kernel.OpCode.RECIP,
            TensorOpCode.ADD: kernel.OpCode.ADD,
            TensorOpCode.MUL: kernel.OpCode.MUL,
            TensorOpCode.POW: kernel.OpCode.POW,
            TensorOpCode.LOG: kernel.OpCode.LOG,
        }
        kerns: list[kernel.Kernel] = []
        for t in tensors:
            if t.code.is_unary_op():
                name, src = self.kernel_generator.generate_unary_kernel(code_map[t.code], t.ndim)

            elif t.code.is_binary_op():
                name, src = self.kernel_generator.generate_binary_kernel(code_map[t.code], t.ndim)

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
                insts.append(InvokeUnaryKernel(kernel.to_kern_name(t.code, t.ndim), t, t.inputs[0]))

            elif t.code.is_binary_op():
                insts.append(AllocateDeviceMemory(t))
                insts.append(InvokeBinaryKernel(kernel.to_kern_name(t.code, t.ndim), t, t.inputs[0], t.inputs[1]))

            elif t.code.is_view_op():
                insts.append(CopyDevicePointer(t.inputs[0], t))

        return insts

    def execute(self, insts: list[Instruction]) -> None:
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
                    params = (inst.src.offset, inst.dst.offset, *inst.src.strides, *inst.dst.strides, inst.src.dev_buffer, inst.dst.dev_buffer)
                    self.kernel_manager.invoke(inst.kern_name, 1, inst.dst.shape, params)

                case InvokeBinaryKernel():
                    params = (
                        inst.srcl.offset,
                        inst.srcr.offset,
                        inst.dst.offset,
                        *inst.srcl.strides,
                        *inst.srcr.strides,
                        *inst.dst.strides,
                        inst.srcl.dev_buffer,
                        inst.srcr.dev_buffer,
                        inst.dst.dev_buffer,
                    )
                    self.kernel_manager.invoke(inst.kern_name, 1, inst.dst.shape, params)
            if dbg:
                print(f"executed: {inst}")


def tensor(arr, requires_grad=False):
    return Tensor(arr)


if __name__ == "__main__":
    t1 = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    t2 = t1[0, 0, 1, 1]
    print(t2.materialize())
    # print(t5)
    # t5.materialize()
    # print(t5)
