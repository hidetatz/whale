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
                i_dev_buff = iss.issue(DeviceBufferAlloc(t.size()))
                i_copy = iss.issue(CopyBufferPythonToDevice(i_py_buff.instid, i_dev_buff.instid))
                insts.extend((i_py_buff, i_dev_buff, i_copy))
                instmap[t] = i_dev_buff.instid

            elif t.op == TensorOp.ADD:
                i_result = iss.issue(DeviceBufferAlloc(t.size()))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("add", 1, t.size(), (l, r, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.MUL:
                i_result = iss.issue(DeviceBufferAlloc(t.size()))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("mul", 1, t.size(), (l, r, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.RECIP:
                i_result = iss.issue(DeviceBufferAlloc(t.size()))
                x = instmap[t.inputs[0]]
                i_inv_kern = iss.issue(InvokeKernel("recip", 1, t.size(), (x, i_result.instid)))
                insts.extend((i_result, i_inv_kern))
                instmap[t] = i_result.instid

            elif t.op == TensorOp.POW:
                i_result = iss.issue(DeviceBufferAlloc(t.size()))
                l = instmap[t.inputs[0]]
                r = instmap[t.inputs[1]]
                i_inv_kern = iss.issue(InvokeKernel("power", 1, t.size(), (l, r, i_result.instid)))
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
                kernel_manager.invoke(
                    inst.kern, inst.grid, inst.block, [self.memory.get(param_id) for param_id in inst.param_ids]
                )

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
            shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.ADD, inputs=inputs, dtype=inputs[0].dtype
        )

    def _backward(self, grad):
        return grad, grad


class Mul(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.MUL, inputs=inputs, dtype=inputs[0].dtype
        )

    def _backward(self, grad):
        return grad * self.inputs[1], grad * self.inputs[0]


class Recip(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.RECIP, inputs=inputs, dtype=inputs[0].dtype
        )

    def _backward(self, grad):
        return grad * (full(self.inputs[0].shape, -1.0) / (self.inputs[0] * self.inputs[0]))


class Pow(BackpropContext):
    def _forward(self, inputs):
        return Tensor(
            shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.POW, inputs=inputs, dtype=inputs[0].dtype
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


class Tensor:
    def __init__(self, data=[], shape=[], stride=[], op=None, creator=None, inputs=[], materialized=False, dtype=None):
        self.data = data
        self.shape = shape
        self.stride = stride
        self.op = op
        self.creator = creator
        self.inputs = inputs
        self.materialized = materialized
        self.dtype = dtype
        self.grad = None
        self.generation = 0

    def size(self):
        return math.prod(self.shape)

    def materialize(self):
        if not self.materialized:
            self.data = Materializer.materialize(self)
            self.materialized = True

        return self

    def add(self, r):
        return _from_calc(Add, [self, r])

    def sub(self, r):
        # l-r is l+(-1*r)
        return _from_calc(Add, [self, _from_calc(Mul, [full(r.shape, -1.0), r])])

    def mul(self, r):
        return _from_calc(Mul, [self, r])

    def div(self, r):
        # l/r = l * (1/r)
        return _from_calc(Mul, [self, _from_calc(Recip, [r])])

    def pow(self, r):
        return _from_calc(Pow, [self, r])

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

    def __matmul__(self, r):
        return _from_calc(MATMUL, [self, r])

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
        return f"Tensor(data: {self.data}, shape: {self.shape}, stride: {self.stride})"

    def str_as_graph(self):
        def f(depth, t):
            indent = "    " * depth
            trail_comma = "," if depth != 0 else ""

            if not t.inputs:
                return f"{indent}Tensor(op:{t.op}, shape:{t.shape}, stride: {t.stride}){trail_comma}"

            input = "[\n" + "\n".join([f(depth + 1, i) for i in t.inputs]) + f"\n{indent}]"
            return f"{indent}Tensor(op:{t.op}, shape:{t.shape}, stride: {t.stride}, input: {input}){trail_comma}"

        return f(0, self)

    def str_as_oneline(self):
        return f"Tensor(op:{self.op}, shape:{self.shape}, stride:{self.stride})"

    def __str__(self):
        return self.str_as_ndarr() if self.materialized else self.str_as_graph()

    def __repr__(self):
        return self.__str__()


def _from_calc(calc, inputs):
    c = calc()
    t = c.forward(inputs)
    t.creator = c
    t.generation = c.generation + 1
    return t


def tensor_with_shape(arr, shape):
    stride = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    return Tensor(
        data=arr, shape=shape, stride=stride, op=TensorOp.CONST, materialized=True, dtype=DType.from_type(type(arr[0]))
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
    wt1 = array(2)
    wt2 = array(2)
    wt3 = wt1**wt2
    print(wt3.materialize())
    # t1 = array([[1, 2, 3], [1, 2, 3]])
    # t2 = array([[4, 5, 6], [4, 5, 6]])
    # t3 = t1 + t2
    # t4 = array([[7, 8, 9], [7, 8, 9]])
    # t5 = t3 * t4

    # t5.backprop()
    # print(t1.grad.materialize())
