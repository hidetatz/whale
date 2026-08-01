import ctypes
import os
import subprocess
import tempfile

import compiler
import executor
import langspec
from kernel import Kernel

class ClangLangSpec(langspec.CCompatibleLangSpec):
    def kern_start(self, name, arg_names, arg_types): return f"void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
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

