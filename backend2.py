import os
import time
from dataclasses import dataclass
from typing import Any
from ctypes import CDLL, byref, c_double, c_float, c_int, c_longlong, c_void_p, cast, pointer, sizeof

class DType:
    pass

class Int32(DType):
    def ctype(self): return c_int

class Int64(DType):
    def ctype(self): return c_longlong

class Float32(DType):
    def ctype(self): return c_float

class Float64(DType):
    def ctype(self): return c_double

int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()

#
# buffer
# 

class CPUBuff:
    def __init__(self, val=None, dtype=None):
        self.val = val
        self.dtype = dtype
        if val and dtype is None:
            assert type(val[0]) is int or type(val[0]) is float
            self.dtype = int64 if type(val[0]) is int else float64

class DevBuff:
    def __init__(self, ptr=None, dtype=None):
        self.ptr = ptr
        self.dtype = dtype

@dataclass
class Kernel:
    name: str
    src: str
    ptr: c_void_p

#
# backend
# 

@dataclass
class Backend:
    pass

class CUDA(Backend):
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")
        self.cuda("cuInit", 0)

        dev = c_int()
        self.cuda("cuDeviceGet", byref(dev), 0)
        self.device_handle = dev.value

        ctx = c_void_p()
        self.cuda("cuCtxCreate", byref(ctx), 0, self.device_handle)
        self.ctx = ctx

    def __del__(self):
        if self.ctx:
            self.cuda("cuCtxDestroy", self.ctx)
            self.ctx = None

    def cuda(self, f, *args):
        fn = getattr(self.libcuda, f)
        result = fn(*args)
        if result != 0: raise RuntimeError(f"{f}: {result}")

    def name(self): return "cuda"

    def memalloc(self, length, dtype):
        ptr = c_void_p()
        self.cuda("cuMemAlloc", byref(ptr), sizeof(dtype.ctype()) * length)
        return ptr

    def free(self, ptr):
        self.cuda("cuMemFree", ptr)

    def memcpy_htod(self, dst, val, length, dtype):
        ctype = dtype.ctype()
        self.cuda("cuMemcpyHtoD", dst, (ctype * length)(*val), sizeof(ctype) * length)

    def memcpy_dtoh(self, src, length, dtype):
        ctype = dtype.ctype()
        out = (ctype * length)()
        self.cuda("cuMemcpyDtoH", out, src, sizeof(ctype) * length)
        return [out[i] for i in range(length)]

    def compile(self, name, src):
        fname = f"/tmp/wh_kern_cuda_{name}_{int(time.time())}"

        # compile cuda src to ptx
        with open(f"{fname}.cu", "w") as f:
            f.write(src)

        result = os.system(f"nvcc --ptx {fname}.cu -o {fname}.ptx")
        if result != 0:
            raise RuntimeError(f"nvcc --ptx failed: {result}")

        with open(f"{fname}.ptx", "rb") as f:
            ptx_src = f.read()

        os.remove(f"{fname}.cu")
        os.remove(f"{fname}.ptx")

        # load ptx as module
        mod = c_void_p()
        self.cuda("cuModuleLoadData", byref(mod), ptx_src)
        ptr = c_void_p()
        self.cuda("cuModuleGetFunction", byref(ptr), mod, name.encode("utf-8"))

        return ptr

    def invoke(self, ptr, grid, block, _params):
        params = (c_void_p * len(_params))()
        for i, p in enumerate(_params):
            if type(p) is DevBuff:
                params[i] = cast(byref(p.ptr), c_void_p)
            elif type(p) is int:
                params[i] = cast(pointer(c_int(p)), c_void_p)
            else:
                raise TypeError(f"unexpected kern param type {type(p)}")

        self.cuda("cuLaunchKernel", ptr, *grid, *block,
            0, # sharedMemBytes
            None, # hStream
            params,
            None, # extra
        )
        self.cuda("cuCtxSynchronize")

cuda = CUDA()

#
# instruction
# 

@dataclass
class MemallocD:
    dst: DevBuff
    length: int

@dataclass
class MemcpyHtoD:
    dst: DevBuff
    src: CPUBuff
    length: int

@dataclass
class MemcpyDtoH:
    dst: CPUBuff
    src: DevBuff
    length: int

@dataclass
class InvokeKernel:
    name: str
    params: list[Any]

@dataclass
class BackendIR:
    backend: Backend
    steps: list[Any]

sum2d_axis0 = """
extern "C" __global__ void sum2d_axis0(
    double* in0,
    double* out,
    int out_size,
    int in0_stride0,
    int in0_stride1,
    int axis0_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size) return;
    int64_t acc = 0;
    for (int axis0_idx = 0; axis0_idx < axis0_size; axis0_idx++) {
        acc += in0[axis0_idx * in0_stride0 + out_idx * in0_stride1];
    }
    out[out_idx] = acc;
} 
"""

fuse1d_add_mul = """
extern "C" __global__ void fuse1d_add_mul(
    double* in0,
    double* in1,
    double* in2,
    double* out,
    int out_size
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size) return;
    int tmp = 0;
    tmp = in0[out_idx] + in1[out_idx];
    tmp = tmp * in2[out_idx];
    out[out_idx] = tmp;
}
"""

class Executor:
    def __init__(self, backend, kernels):
        self.backend = backend
        self.kernels = {k.name: k for k in kernels}

    def execute(self, steps):
        for step in steps:
            match step:
                case MemallocD():
                    step.dst.ptr = self.backend.memalloc(step.length, step.dst.dtype)
                case MemcpyHtoD():
                    self.backend.memcpy_htod(step.dst.ptr, step.src.val, step.length, step.dst.dtype)
                case MemcpyDtoH():
                    step.dst.val = self.backend.memcpy_dtoh(step.src.ptr, step.length, step.dst.dtype)
                case InvokeKernel():
                    kern = self.kernels[step.name]
                    self.backend.invoke(kern.ptr, (1, 1, 1), (32, 1, 1), step.params)


if __name__ == "__main__":
    # a = Tensor([[1, 2, 3], [1, 2, 3]])
    # b = Tensor([4, 5, 6])
    # c = Tensor(10)
    # d = a.sum(axis=0)
    # e = b + d
    # f = e * c
    # f.materialize()
    a = CPUBuff([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    a_dev = DevBuff(dtype=a.dtype)

    b = CPUBuff([4.0, 5.0, 6.0])
    b_dev = DevBuff(dtype=b.dtype)

    c = CPUBuff([10.0, 10.0, 10.0])
    c_dev = DevBuff(dtype=c.dtype)

    d_dev = DevBuff(dtype=a_dev.dtype)
    f_dev = DevBuff(dtype=d_dev.dtype)
    f = CPUBuff(dtype=f_dev.dtype)

    ir = BackendIR(
        backend=cuda,
        steps=[
            MemallocD(dst=a_dev, length=6),
            MemcpyHtoD(dst=a_dev, src=a, length=6),
            MemallocD(dst=b_dev, length=3),
            MemcpyHtoD(dst=b_dev, src=b, length=3),
            MemallocD(dst=c_dev, length=3),
            MemcpyHtoD(dst=c_dev, src=c, length=3),
            MemallocD(dst=d_dev, length=3),
            InvokeKernel(name="sum2d_axis0", params=[a_dev, d_dev, 3, 3, 1, 2]),
            MemallocD(dst=f_dev, length=3),
            InvokeKernel(name="fuse1d_add_mul", params=[b_dev, d_dev, c_dev, f_dev, 3]),
            MemcpyDtoH(src=f_dev, dst=f, length=3),
        ],
    )

    kernels = [
        Kernel("sum2d_axis0", sum2d_axis0, cuda.compile("sum2d_axis0", sum2d_axis0)),
        Kernel("fuse1d_add_mul", fuse1d_add_mul, cuda.compile("fuse1d_add_mul", fuse1d_add_mul)),
    ]

    e = Executor(ir.backend, kernels)
    e.execute(ir.steps)

    print(f.val)
