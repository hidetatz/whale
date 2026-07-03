import ctypes
import os
import subprocess
import tempfile

import dtype

class ClangC:
    def __init__(self):
        self.kerns = {}

    @classmethod
    def is_gpu(cls): return False

    def typename(self, dt):
        if dt == dtype.int32: return "int32_t"
        elif dt == dtype.int64: return "int64_t"
        elif dt == dtype.float32: return "float"
        elif dt == dtype.float64: return "double"
        else: raise RuntimeError(f"unknown dtype: {dt}")

    # lang settings
    def import_lib(self, lib): return f"#include <{lib}>"
    def default_library(self): return ["stdint.h", "math.h"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
    def kern_end(self): return "}"
    def loop_start(self, index, start, end, step): return f"for (int {index} = {start}; {index} < {end}; {index} += {step}) {{"
    def loop_end(self): return "}"
    def index(self, a, idx): return f"{a}[{idx}]"
    def init(self, dt, l, r): return f"{self.typename(dt)} {l} = {r};"
    def assign(self, l, r): return f"{l} = {r};"
    def neg(self, a): return f"-({a})"
    def pow(self, a): return f"pow({a}, 2)"
    def sin(self, a): return f"sin({a})"
    def cos(self, a): return f"cos({a})"
    def exp(self, a): return f"exp({a})"
    def log(self, a): return f"log({a})"
    def sqrt(self, a): return f"sqrt({a})"
    def add(self, l, r): return f"{l} + {r}"
    def sub(self, l, r): return f"{l} - {r}"
    def mul(self, l, r): return f"{l} * {r}"
    def truediv(self, l, r): return f"{l} / {r}"

    # exec settings
    def compile(self, name, code):
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f: so = f.name
        subprocess.run(["clang", "-x", "c", "-", "-O2", "-shared", "-fPIC", "-o", so, "-lm"], input=code, check=True, text=True)
        kern = ctypes.CDLL(so)
        os.remove(so)  # loaded, ok to delete the file
        self.kerns[name] = kern

    def execute(self, name, param_buffs):
        ptr = self.kerns[name]
        kern = getattr(ptr, name)
        kern.argtypes = [ctypes.POINTER(buf.cpu.dtype.ctype()) for buf in param_buffs]
        kern.restype = None  # void
        params = [(buf.cpu.dtype.ctype() * len(buf.cpu.val))(*buf.cpu.val) for buf in param_buffs]
        kern(*params)
        param_buffs[0].cpu.val[:] = list(params[0])

