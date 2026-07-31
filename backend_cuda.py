import os
import subprocess
import tempfile
from ctypes import byref, cast, sizeof, c_void_p, c_int
from ctypes import CDLL

import compiler
import executor
import langspec
import dtype
from kernel import Kernel

class CUDALangSpec(langspec.HighLevelLangSpec):
    def typename(self, dt):
        if dt == dtype.int32: return "int32_t"
        elif dt == dtype.int64: return "int64_t"
        elif dt == dtype.float32: return "float"
        elif dt == dtype.float64: return "double"
        else: raise RuntimeError(f"unknown dtype: {dt}")
    def import_lib(self, lib): return f"#include <{lib}>"
    def default_library(self): return ["stdint.h", "math.h"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"extern \"C\" __global__ void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
    def kern_end(self): return "}"
    def sequential_loop_start(self, index, start, end, step): return f"for (int {index} = {start}; {index} < {end}; {index} += {step}) {{"
    def parallel_loop_start(self, index, start, end, step): raise RuntimeError("cuda backend cannot handle parallel loop: use gpu_blocks/threads instead!")
    def vectorized_loop_start(self, index, start, end, step): raise RuntimeError("cuda backend cannot handle vectorized loop: use gpu_blocks/threads instead!")
    def unrolled_loop_start(self, index, start, end, step, factor):
        return [
            f"#pragma unroll {factor}",
            self.sequential_loop_start(index, start, end, step),
        ]
    def gpu_block_index(self, index, start, end, step, idx): return self.init(dtype.int64, index, f"blockIdx.{idx}")
    def gpu_thread_index(self, index, start, end, step, idx): return self.init(dtype.int64, index, f"threadIdx.{idx}")
    def loop_end(self): return "}"
    def guard(self, cond): return [f"if ({cond}) {{", "return;", "}"]
    def greater_than(self, l, r): return f"{l} <= {r}"
    def index(self, a, idx): return f"{a}[{idx}]"
    def init(self, dt, l, r): return f"{self.typename(dt)} {l} = {r};"
    def assign(self, l, r): return f"{l} = {r};"
    def neg(self, a): return f"-({a})"
    def sin(self, a): return f"sin({a})"
    def cos(self, a): return f"cos({a})"
    def exp(self, a): return f"exp({a})"
    def log(self, a): return f"log({a})"
    def sqrt(self, a): return f"sqrt({a})"
    def add(self, l, r): return f"{l} + {r}"
    def sub(self, l, r): return f"{l} - {r}"
    def mul(self, l, r): return f"{l} * {r}"
    def truediv(self, l, r): return f"{l} / {r}"
    def pow(self, l, r): return f"pow({l}, {r})"

class CUDA:
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")
        self.exec("cuInit", 0)

        dev = c_int()
        self.exec("cuDeviceGet", byref(dev), 0)
        self.device_handle = dev.value

        ctx = c_void_p()
        self.exec("cuCtxCreate", byref(ctx), 0, self.device_handle)
        self.ctx = ctx

    def exec(self, f, *args):
        fn = getattr(self.libcuda, f)
        result = fn(*args)
        if result != 0: raise RuntimeError(f"{f}: {result}")

    def __del__(self):
        if self.ctx:
            self.exec("cuCtxDestroy", self.ctx)
            self.ctx = None

class CUDACompiler(compiler.Compiler):
    def __init__(self, cuda):
        self.cuda = cuda

    # exec settings
    def compile(self, name, code):
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as f: ptx = f.name
        subprocess.run(["nvcc", "-ptx", "-x", "cu", "-", "-o", ptx], input=code, check=True, text=True)
        with open(f"{ptx}", "rb") as f: ptx_src = f.read()
        os.remove(ptx)
        # load ptx as module
        mod = c_void_p()
        self.cuda.exec("cuModuleLoadData", byref(mod), ptx_src)
        ptr = c_void_p()
        self.cuda.exec("cuModuleGetFunction", byref(ptr), mod, name.encode("utf-8"))
        return Kernel(ptr)

class CUDAExecutor(executor.GPUExecutor):
    def __init__(self, cuda):
        self.cuda = cuda

    def execute(self, kern: Kernel, params: list[buffer.Buffer], grid: tuple[int, int, int], block: tuple[int, int, int]):
        kern_ptr = kern.bin
        c_params = (c_void_p * len(params))()
        for i, p in enumerate(params):
            c_params[i] = cast(byref(p.dev.ptr), c_void_p)

        self.cuda.exec("cuLaunchKernel", kern_ptr, *grid, *block,
            0, # sharedMemBytes
            None, # hStream
            c_params,
            None, # extra
        )
        self.cuda.exec("cuCtxSynchronize")

    def memalloc(self, length, ctype):
        ptr = c_void_p()
        self.cuda.exec("cuMemAlloc", byref(ptr), sizeof(ctype) * length)
        return ptr

    def free(self, ptr):
        self.cuda.exec("cuMemFree", ptr)

    def memcpy_htod(self, dst, src, length, ctype):
        self.cuda.exec("cuMemcpyHtoD", dst, (ctype * length)(*src), sizeof(ctype) * length)

    def memcpy_dtoh(self, src, length, ctype):
        out = (ctype * length)()
        self.cuda.exec("cuMemcpyDtoH", out, src, sizeof(ctype) * length)
        return [out[i] for i in range(length)]
