import collections
from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import time

import cuda
import device

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

        backend = Backend.detect()
        if backend == Backend.CUDA:
            self.renderer = cuda.Renderer()
            self.allocator = cuda.Allocator()
            self.kernel_manager = cuda.KernelManager()

        self.initialized = True

    @classmethod
    def materialize(cls, t):
        self = cls() # get materializer singleton instance

        tensors = self.linearize(t)

        # generate and load kernels
        kernel_srcs = self.generate_kernels(tensors)
        self.kernel_manager.load(kernel_srcs)

        # compile tensors into instructions
        instructions = self.generate_instructions(tensors)

        return self.executor.execute(instructions, self.allocator, self.kernel_manager).value

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

        return kerns

    def generate_instructions(self, tensors):
        instmap = {}
        insts = []
        gen = InstructionGenerator()

        for i, t in enumerate(reversed(tensors)):
            if t.op == TensorOp.CONST:
                insts.append(gen.device_buffer_alloc(t.size()))
                insts.append(gen.copy_buffer_python_to_device(device.PythonBuffer(t.data), insts[-1].instid))
                instmap[t] = insts[-1].instid

            elif t.op == TensorOp.ADD:
                insts.append(gen.device_buffer_alloc(t.size()))
                insts.append(gen.invoke_kernel("add", 1, t.size(), (instmap[t.inputs[0]], instmap[t.inputs[1]], insts[-1].instid), insts[-1].instid))
                instmap[t] = insts[-1].instid

            elif t.op == TensorOp.MUL:
                insts.append(gen.device_buffer_alloc(t.size()))
                insts.append(gen.invoke_kernel("mul", 1, t.size(), (instmap[t.inputs[0]], instmap[t.inputs[1]], insts[-1].instid), insts[-1].instid))
                instmap[t] = insts[-1].instid

            if i == len(tensors) - 1:
                insts.append(gen.copy_buffer_device_to_python(insts[-1].instid, device.PythonBuffer([])))
                insts.append(gen.return_value_to_python(insts[-1].instid))

        return insts

class Mnemonic(Enum):
    DEVICE_BUFFER_ALLOC = auto()
    COPY_BUFFER_DEVICE_TO_PYTHON = auto()
    COPY_BUFFER_PYTHON_TO_DEVICE = auto()
    INVOKE_KERNEL = auto()
    RETURN_VALUE_TO_PYTHON = auto()

class InstructionGenerator:
    def __init__(self):
        self.current_id = 0

    def generate(self, mnemonic, **kwargs):
        self.current_id += 1
        return Instruction(self.current_id, mnemonic, **kwargs)

    def python_buffer_alloc(self, length):
        return self.generate(Mnemonic.PYTHON_BUFFER_ALLOC)

    def device_buffer_alloc(self, length):
        return self.generate(Mnemonic.DEVICE_BUFFER_ALLOC, length=length)

    def copy_buffer_python_to_device(self, py_buff, gpu_buff_id):
        return self.generate(Mnemonic.COPY_BUFFER_PYTHON_TO_DEVICE, py_buff=py_buff, gpu_buff_id=gpu_buff_id)

    def copy_buffer_device_to_python(self, gpu_buff_id, py_buff):
        return self.generate(Mnemonic.COPY_BUFFER_DEVICE_TO_PYTHON, gpu_buff_id=gpu_buff_id, py_buff=py_buff)

    def invoke_kernel(self, kern_name, grid, block, input_ids, output_id):
        return self.generate(Mnemonic.INVOKE_KERNEL, kern_name=kern_name, grid=grid, block=block, input_ids=input_ids, output_id=output_id)

    def return_value_to_python(self, buff_id):
        return self.generate(Mnemonic.RETURN_VALUE_TO_PYTHON, buff_id=buff_id)

class Instruction:
    def __init__(self, instid, mnemonic, **kwargs):
        self.instid = instid
        self.mnemonic = mnemonic
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        input = "()" if not hasattr(self, "input") else f"{self.input}" if type(self.input) == tuple else f"({self.input})"
        return f"<{self.instid}:{self.mnemonic}{input}>"

    def __repr__(self):
        return self.__str__()

class Executor:
    class MemoryManager:
        def __init__(self):
            self.memory = {"python": {}, "device": {}}

        def allocate_python(self, instid, buf):
            self.memory["python"][instid] = buf

        def allocate_device(self, instid, buf):
            self.memory["device"][instid] = buf

        def get_python_buff(self, instid):
            return self.memory["python"][instid]

        def get_device_buff(self, instid):
            return self.memory["device"][instid]

    def __init__(self):
        self.memory = self.MemoryManager()

    def execute(self, instructions, allocator, kernel_manager):
        for inst in instructions:
            if inst.mnemonic == Mnemonic.DEVICE_BUFFER_ALLOC:
                gpu_buff = allocator.allocate(inst.length)
                self.memory.allocate_device(inst.instid, gpu_buff)

            elif inst.mnemonic == Mnemonic.COPY_BUFFER_PYTHON_TO_DEVICE:
                gpu_buff = self.memory.get_device_buff(inst.gpu_buff_id)
                allocator.copy_to_device(inst.py_buff, gpu_buff)
                self.memory.allocate_device(inst.instid, gpu_buff)

            elif inst.mnemonic == Mnemonic.COPY_BUFFER_DEVICE_TO_PYTHON:
                gpu_buff = self.memory.get_device_buff(inst.gpu_buff_id)
                py_buff = allocator.copy_from_device(gpu_buff)
                self.memory.allocate_python(inst.instid, py_buff)

            elif inst.mnemonic == Mnemonic.INVOKE_KERNEL:
                kernel_manager.invoke(inst.kern_name, inst.grid, inst.block, [self.memory.get_device_buff(instid) for instid in inst.input_ids])
                out = self.memory.get_device_buff(inst.output_id)
                self.memory.allocate_device(inst.instid, out)

            elif inst.mnemonic == Mnemonic.RETURN_VALUE_TO_PYTHON:
                return self.memory.get_python_buff(inst.buff_id)

class Calculation:
    def __call__(self, inputs):
        self.inputs = inputs
        self.generation = max([i.generation for i in inputs])
        out = self.forward(inputs)
        self.output = out
        return out

    def backward(self, grad): raise NotImplementedError()

class Add(Calculation):
    def forward(self, inputs):
        return Tensor(shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.ADD, inputs=inputs, dtype=inputs[0].dtype)

    def backward(self, grad):
        return grad, grad

class Mul(Calculation):
    def forward(self, inputs):
        self.l = inputs[0]
        self.r = inputs[1]
        return Tensor(shape=inputs[0].shape, stride=inputs[0].stride, op=TensorOp.MUL, inputs=inputs, dtype=inputs[0].dtype)

    def backward(self, grad):
        return grad*self.r, grad*self.l

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

    def relu(self):
        return _from_calc(MAXIMUM, [self, zeros_like(self.shape)])

    def __add__(self, r):
        return _from_calc(Add, [self, r])

    def __mul__(self, r):
        return _from_calc(Mul, [self, r])

    def __matmul__(self, r):
        return _from_calc(MATMUL, [self, r])

    def str_ndarr(self):
        if len(self.shape) == 0:
            return f"{self.data[0]}"

        if 0 in self.shape:
            return f"[]"

        if len(self.shape) == 1:
            return f"[{', '.join(map(str, self.data))}]"

        # todo: implement
        return f"[{', '.join(map(str, self.data))}] (shape: {self.shape}, stride: {self.stride})"

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
            gy = c.output
            gxs = c.backward(gy)
            if not isinstance(gxs, tuple):
                gxs = gxs,

            for x, gx in zip(c.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    f(x.creator)

    def backward(self):
        self.backprop()

    def __str__(self):
        if self.materialized:
            return f"Tensor({self.str_ndarr()})"

        else:
            if self.op == TensorOp.ADD:
                return f"{self.inputs[0]} + {self.inputs[1]}"

            if self.op == TensorOp.MUL:
                return f"{self.inputs[0]} * {self.inputs[1]}"

            raise NotImplementedError()

    def __repr__(self):
        return self.__str__()
        
    def debug(self):
        return f"data: {self.data}, shape: {self.shape}, stride: {self.stride}, op: {self.op}, materialized: {self.materialized}, dtype: {self.dtype}"

def _from_calc(calc, inputs):
    c = calc()
    t = c(inputs)
    t.creator = c
    t.generation = c.generation + 1
    return t

def tensor_with_shape(arr, shape):
    stride = [math.prod(shape[i+1:]) for i in range(len(shape))]
    return Tensor(data=arr, shape=shape, stride=stride, op=TensorOp.CONST, materialized=True, dtype=DType.from_type(type(arr[0])))

def tensor(arr):
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
                f(elem, dim+1)

        f(arr, 0)
        return tensor_with_shape(data, shape)

def array(arr):
    return tensor(arr)

def full(shape, value):
    return tensor_with_shape([value] * math.prod(shape), shape)

def ones_like(t):
    return full(t.shape, 1)

def zeros_like(t):
    return full(t.shape, 0)

t1 = tensor([[1, 2, 3], [1, 2, 3]])
t2 = tensor([[4, 5, 6], [4, 5, 6]])
t3 = t1 + t2
t4 = tensor([[7, 8, 9], [7, 8, 9]])
t5 = t3 * t4

t5.backprop()
print(t1.grad.materialize())
