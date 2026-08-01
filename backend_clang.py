import ctypes
import os
import subprocess
import tempfile

import compiler
import executor
import langspec
import dtype
from kernel import Kernel

class ClangLangSpec(langspec.HighLevelLangSpec):
    def typename(self, dt):
        if dt == dtype.int32: return "int32_t"
        elif dt == dtype.int64: return "int64_t"
        elif dt == dtype.float32: return "float"
        elif dt == dtype.float64: return "double"
        else: raise RuntimeError(f"unknown dtype: {dt}")
    def import_lib(self, lib): return f"#include <{lib}>"
    def default_library(self): return ["stdint.h", "math.h"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
    def kern_end(self): return "}"
    def sequential_loop_start(self, index, start, end, step): return f"for (int {index} = {start}; {index} < {end}; {index} += {step}) {{"
    def parallel_loop_start(self, index, start, end, step):
        return [
            "#pragma omp parallel for",
            self.sequential_loop_start(index, start, end, step),
        ]
    def vectorized_loop_start(self, index, start, end, step):
        return [
            "#pragma clang loop vectorize(enable)",
            self.sequential_loop_start(index, start, end, step),
        ]
    def unrolled_loop_start(self, index, start, end, step, factor):
        return [
            f"#pragma clang loop unroll_count({factor})",
            self.sequential_loop_start(index, start, end, step),
        ]
    def gpu_block_index(self, index, start, end, step, idx): raise RuntimeError("clang backend cannot handle gpu blocks!")
    def gpu_thread_index(self, index, start, end, step, idx): raise RuntimeError("clang backend cannot handle gpu blocks!")
    def loop_end(self): return "}"
    def if_start(self, cond): return f"if ({cond}) {{"
    def if_end(self): return "}"
    def return_function(self): return "return;"
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

class ClangCompiler(compiler.Compiler):
    def compile(self, name: str, code: str):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f: so = f.name
        subprocess.run(["clang", "-x", "c", "-", "-O2", "-shared", "-fPIC", "-fopenmp", "-o", so, "-lm"], input=code, check=True, text=True)
        kern = ctypes.CDLL(so)
        os.remove(so)  # loaded, ok to delete the file
        kern_fn = getattr(kern, name)
        return Kernel(kern_fn)

class ClangExecutor(executor.CPUExecutor):
    def execute(self, kern: Kernel, params: list[buffer.Buffer]):
        kern_fn = kern.bin
        kern_fn.argtypes = [ctypes.POINTER(p.dtype.ctype()) for p in params]
        kern_fn.restype = None  # void
        c_params = [(buf.dtype.ctype() * len(buf.cpu.val))(*buf.cpu.val) for buf in params]
        kern_fn(*c_params)
        params[0].cpu.val[:] = list(c_params[0])

