import collections
from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import time

import cuda
import device

dbg = os.getenv("WHALE_DEBUG", "") != ""


class Backend(Enum):
    PYTHON = auto()
    CUDA = auto()

    @classmethod
    def detect(cls):
        be = os.environ.get("WHALE_BACKEND")
        if be is None:

            def cmd_avail(cmd):
                return os.system(f"command -v {cmd} > /dev/null") == 0

            # auto detect
            if cmd_avail("nvcc"):
                return cls.CUDA

            return cls.PYTHON

        if be == "CUDA":
            return cls.CUDA

        return cls.PYTHON


class Materializer:
    # materializer is singleton for caching some info
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return

        self.executor = Executor()

        self.backend = Backend.detect()
        if self.backend == Backend.CUDA:
            self.renderer = cuda.Renderer()
            self.allocator = cuda.Allocator()
            self.kernel_manager = cuda.KernelManager()

        self.initialized = True

    @classmethod
    def materialize(cls, t):
        self = cls()  # get materializer singleton instance

        if dbg:
            print("=== materialization start ===")
            print("=== backend")
            print(self.backend)

            print("=== tensor graph")
            print(t.str_as_graph())

        tensors = self.linearize(t)

        if dbg:
            print("=== linearized tensors")
            print("\n".join([f"{t.str_as_oneline()}" for t in tensors]))

        # generate and load kernels
        kernel_srcs = self.generate_kernels(tensors)

        if dbg:
            print("=== kernels")
            print("\n\n".join([f"{kern_src.src}" for kern_src in kernel_srcs]))

        self.kernel_manager.load(kernel_srcs)

        # compile tensors into instructions
        instructions = self.generate_instructions(tensors)
        if dbg:
            print("=== instructions")
            print("\n".join([f"{inst}" for inst in instructions]))

        result = self.executor.execute(instructions, self.allocator, self.kernel_manager).value

        if dbg:
            print("=== materialization successfully finished ===")

        return result

    def linearize(self, t):
        tensors = []
        seen = []

        def dfs(_t):
            if _t in seen:
                return

            seen.append(_t)
            tensors.append(_t)

            for i in _t.inputs:
                dfs(i)

        dfs(t)
        tensors.reverse()
        return tensors

    def generate_kernels(self, tensors):
        ops = set([t.op for t in tensors])
        ops.remove(TensorOp.CONST)
        kerns = []
        for op in ops:
            if op == TensorOp.ADD:
                kerns.append(device.KernelSrc(self.renderer.render_kern_add(), "add"))
            elif op == TensorOp.MUL:
                kerns.append(device.KernelSrc(self.renderer.render_kern_mul(), "mul"))
            elif op == TensorOp.RECIP:
                kerns.append(device.KernelSrc(self.renderer.render_kern_recip(), "recip"))
            elif op == TensorOp.POW:
                kerns.append(device.KernelSrc(self.renderer.render_kern_pow(), "power"))

        return kerns

    def generate_instructions(self, tensors):
        instmap = {}
        insts = []
        iss = InstIssuer()

        for i, t in enumerate(tensors):
            if t.op == TensorOp.CONST:
                i_py_buff = iss.issue(PythonBufferAlloc(t.data))
                i_dev_buff = iss.issue(DeviceBufferAlloc(t.size))
                i_copy = iss.issue(CopyBufferPythonToDevice(i_py_buff.instid, i_dev_buff.instid))
                insts.extend((i_py_buff, i_dev_buff, i_copy))
                instmap[t] = i_dev_buff.instid

            elif t.op == TensorOp.ADD:
                i_result = iss.issue(DeviceBufferAlloc(t.size))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("add", 1, t.size, (l, r, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.MUL:
                i_result = iss.issue(DeviceBufferAlloc(t.size))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("mul", 1, t.size, (l, r, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.RECIP:
                i_result = iss.issue(DeviceBufferAlloc(t.size))
                x = instmap[t.inputs[0]]
                i_inv_kern = iss.issue(InvokeKernel("recip", 1, t.size, (x, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.POW:
                i_result = iss.issue(DeviceBufferAlloc(t.size))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("power", 1, t.size, (l, r, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            if i == len(tensors) - 1:
                i_py_buff = iss.issue(PythonBufferAlloc(None))
                i_last = instmap[t]
                i_copy = iss.issue(CopyBufferDeviceToPython(i_last, i_py_buff.instid))
                i_term = iss.issue(Terminate(i_py_buff.instid))
                insts.extend((i_py_buff, i_copy, i_term))

        return insts


class InstIssuer:
    def __init__(self):
        self.current_id = 0

    def issue(self, inst):
        self.current_id += 1
        inst.set_id(self.current_id)
        return inst


class Inst:
    def set_id(self, instid):
        self.instid = instid

    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != "instid")
        return f"<{self.instid}:{self.__class__.__name__}{f"({params})"}>"

    def __repr__(self):
        return self.__str__()


@dataclass(repr=False)
class PythonBufferAlloc(Inst):
    data: list


@dataclass(repr=False)
class DeviceBufferAlloc(Inst):
    length: int


@dataclass(repr=False)
class CopyBufferPythonToDevice(Inst):
    py_buff_id: int
    gpu_buff_id: int


@dataclass(repr=False)
class CopyBufferDeviceToPython(Inst):
    gpu_buff_id: int
    py_buff_id: int


@dataclass(repr=False)
class InvokeKernel(Inst):
    kern: str
    grid: any
    block: any
    param_ids: tuple


@dataclass
class Terminate(Inst):
    ret_id: int


class Executor:
    class MemoryManager:
        def __init__(self):
            self.memory = {}

        def set(self, addr, buf):
            self.memory[addr] = buf

        def get(self, addr):
            return self.memory[addr]

    def __init__(self):
        self.memory = self.MemoryManager()

    def execute(self, instructions, allocator, kernel_manager):
        for inst in instructions:
            if isinstance(inst, PythonBufferAlloc):
                self.memory.set(inst.instid, device.PythonBuffer(inst.data))

            elif isinstance(inst, DeviceBufferAlloc):
                self.memory.set(inst.instid, allocator.allocate(inst.length))

            elif isinstance(inst, CopyBufferPythonToDevice):
                py_buff = self.memory.get(inst.py_buff_id)
                gpu_buff = self.memory.get(inst.gpu_buff_id)
                allocator.copy_to_device(py_buff, gpu_buff)

            elif isinstance(inst, CopyBufferDeviceToPython):
                gpu_buff = self.memory.get(inst.gpu_buff_id)
                py_buff = self.memory.get(inst.py_buff_id)
                allocator.copy_from_device(gpu_buff, py_buff)

            elif isinstance(inst, InvokeKernel):
                kernel_manager.invoke(inst.kern, inst.grid, inst.block, [self.memory.get(param_id) for param_id in inst.param_ids])

            elif isinstance(inst, Terminate):
                return self.memory.get(inst.ret_id)


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


class Tensor:
    def __init__(self, data=[], shape=[], strides=[], offset=0, op=None, creator=None, inputs=[], materialized=False, dtype=None):
        self.data = data
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.op = op
        self.creator = creator
        self.inputs = inputs
        self.materialized = materialized
        self.dtype = dtype
        self.grad = None
        self.generation = 0

    @property
    def size(self):
        return math.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def materialize(self):
        if not self.materialized:
            self.data = Materializer.materialize(self)
            self.materialized = True

        return self

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

        newshape = [s for s in newshape if s != -1]
        newstrides = [s for s in newstrides if s != -1]
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
        return self.data

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
        def f(depth, t):
            indent = "    " * depth
            trail_comma = "," if depth != 0 else ""

            if not t.inputs:
                return f"{indent}Tensor(op:{t.op}, offset: {self.offset}, shape:{t.shape}, strides: {t.strides}){trail_comma}"

            input = "[\n" + "\n".join([f(depth + 1, i) for i in t.inputs]) + f"\n{indent}]"
            return f"{indent}Tensor(op:{t.op}, offset: {t.offset}, shape:{t.shape}, strides: {t.strides}, input: {input}){trail_comma}"

        return f(0, self)

    def str_as_oneline(self):
        return f"Tensor(op:{self.op}, shape:{self.shape}, strides:{self.strides})"

    def __str__(self):
        return self.str_as_ndarr() if self.materialized else self.str_as_graph()

    def __repr__(self):
        return self.__str__()


def _from_calc(calc, inputs):
    t = calc.forward(inputs)
    t.creator = calc
    t.generation = calc.generation + 1
    return t


def tensor_with_shape(arr, shape):
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    return Tensor(
        data=arr,
        shape=shape,
        strides=strides,
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
        return tensor_with_shape(data, shape)


def full(shape, value):
    return tensor_with_shape([value] * math.prod(shape), shape)


def ones_like(t):
    return full(t.shape, 1.0)


def zeros_like(t):
    return full(t.shape, 0.0)


if __name__ == "__main__":
    t1 = array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
    t2 = t1[0, 1]
    t2.materialize()
    # t1 = array([[1, 2, 3], [1, 2, 3]])
    # t2 = array([[4, 5, 6], [4, 5, 6]])
    # t3 = t1 + t2
    # t4 = array([[7, 8, 9], [7, 8, 9]])
    # t5 = t3 * t4

    # t5.backprop()
    # print(t1.grad.materialize())
