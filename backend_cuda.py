import os
import subprocess
import tempfile
from ctypes import byref, cast, sizeof, c_void_p, c_int
from ctypes import CDLL

import dtype

class CUDA:
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")
        self.cuda("cuInit", 0)

        dev = c_int()
        self.cuda("cuDeviceGet", byref(dev), 0)
        self.device_handle = dev.value

        ctx = c_void_p()
        self.cuda("cuCtxCreate", byref(ctx), 0, self.device_handle)
        self.ctx = ctx

        self.kerns = {}

    @classmethod
    def is_gpu(cls): return True

    def typename(self, dt):
        if dt == dtype.int32: return "int32_t"
        elif dt == dtype.int64: return "int64_t"
        elif dt == dtype.float32: return "float"
        elif dt == dtype.float64: return "double"
        else: raise RuntimeError(f"unknown dtype: {dt}")

    def cuda(self, f, *args):
        fn = getattr(self.libcuda, f)
        result = fn(*args)
        if result != 0: raise RuntimeError(f"{f}: {result}")

    def __del__(self):
        if self.ctx:
            self.cuda("cuCtxDestroy", self.ctx)
            self.ctx = None

    # lang settings
    def import_lib(self, lib): return f"#include <{lib}>"
    def default_library(self): return ["stdint.h", "math.h"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"extern \"C\" __global__ void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
    def kern_end(self): return "}"
    def loop_start(self, index, start, end, step): return f"for (int {index} = {start}; {index} < {end}; {index} += {step}) {{"
    def loop_end(self): return "}"
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

    # exec settings
    def compile(self, name, code):
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as f: ptx = f.name
        subprocess.run(["nvcc", "-ptx", "-x", "cu", "-", "-o", ptx], input=code, check=True, text=True)
        with open(f"{ptx}", "rb") as f: ptx_src = f.read()
        os.remove(ptx)
        # load ptx as module
        mod = c_void_p()
        self.cuda("cuModuleLoadData", byref(mod), ptx_src)
        ptr = c_void_p()
        self.cuda("cuModuleGetFunction", byref(ptr), mod, name.encode("utf-8"))
        self.kerns[name] = ptr

    def execute(self, name, param_buffs):
        kern_ptr = self.kerns[name]
        params = (c_void_p * len(param_buffs))()
        for i, p in enumerate(param_buffs):
            params[i] = cast(byref(p.dev.ptr), c_void_p)

        grid = (1, 1, 1)
        block = (32, 1, 1)

        self.cuda("cuLaunchKernel", kern_ptr, *grid, *block,
            0, # sharedMemBytes
            None, # hStream
            params,
            None, # extra
        )
        self.cuda("cuCtxSynchronize")

        # param_buffs[0].cpu.val[:] = list(params[0])

    def memalloc(self, length, ctype):
        ptr = c_void_p()
        self.cuda("cuMemAlloc", byref(ptr), sizeof(ctype) * length)
        return ptr

    def free(self, ptr):
        self.cuda("cuMemFree", ptr)

    def memcpy_htod(self, dst, src, length, ctype):
        self.cuda("cuMemcpyHtoD", dst, (ctype * length)(*src), sizeof(ctype) * length)

    def memcpy_dtoh(self, src, length, ctype):
        out = (ctype * length)()
        self.cuda("cuMemcpyDtoH", out, src, sizeof(ctype) * length)
        return [out[i] for i in range(length)]
