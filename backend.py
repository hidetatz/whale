import os
from abc import ABC, abstractmethod

import buffer
import sched
from codegen import CodeGenerator, HighLevelLangCodeGenerator
from executor import Executor, CPUExecutor, GPUExecutor
from kernel import Kernel
from backend_clang import ClangLangSpec, ClangCompiler, ClangExecutor
from backend_cuda import CUDALangSpec, CUDACompiler, CUDAExecutor, CUDA
from backend_python import PythonLangSpec, PythonCompiler, PythonExecutor

class Backend(ABC):
    def __str__(self): return self.__class__.__name__
    def __repr__(self): return self.__class__.__name__

    def __init__(self, codegenerator: CodeGenerator, compiler: Compiler, executor: Executor):
        self.codegenerator = codegenerator
        self.compiler = compiler
        self.executor = executor

    def codegen(self, kern_name, func, schedule, inputs): return self.codegenerator.codegen(kern_name, func, schedule, inputs)
    def compile(self, name: str, code: str): return self.compiler.compile(name, code)

    @abstractmethod
    def invoke_kernel(self, kern: Kernel, params: list[buffer.Buffer]): pass

class CPUBackend(Backend):
    def __init__(self, codegenerator: CodeGenerator, compiler: Compiler, executor: CPUExecutor):
        super().__init__(codegenerator, compiler, executor)

    def invoke_kernel(self, schedule: sched.Schedule, kern: Kernel, params: list[buffer.Buffer]):
        for i, p in enumerate(params):
            if p.cpu is None:
                p.cpu = buffer.CPUBuff([0] * p.length)

        self.executor.execute(kern, params)

class GPUBackend(Backend):
    def __init__(self, codegenerator: CodeGenerator, compiler: Compiler, executor: GPUExecutor):
        super().__init__(codegenerator, compiler, executor)

    def invoke_kernel(self, schedule: sched.Schedule, kern: Kernel, params: list[buffer.Buffer]):
        e = self.executor
        for i, p in enumerate(params):
            if p.dev.ptr is None:
                # memalloc
                p.dev.ptr = e.memalloc(p.length, p.dtype.ctype())

            # memcpy
            if i != 0 and p.cpu is not None: e.memcpy_htod(p.dev.ptr, p.cpu.val, p.length, p.dtype.ctype())

        dim = {"x": 0, "y": 1, "z": 2}
        grid = [1, 1, 1]
        block = [1, 1, 1]

        for l in schedule.loops:
            if isinstance(l.exec, sched.GPUBlock):
                grid[dim[l.exec.index]] = l.lv.extent
            elif isinstance(l.exec, sched.GPUThread):
                block[dim[l.exec.index]] = l.lv.extent

        e.execute(kern, params, tuple(grid), tuple(block))

class PythonBackend(CPUBackend):
    def __init__(self):
        super().__init__(HighLevelLangCodeGenerator(PythonLangSpec()), PythonCompiler(), PythonExecutor())

class ClangBackend(CPUBackend):
    def __init__(self):
        super().__init__(HighLevelLangCodeGenerator(ClangLangSpec()), ClangCompiler(), ClangExecutor())

class CUDABackend(GPUBackend):
    def __init__(self):
        c = CUDA()
        super().__init__(HighLevelLangCodeGenerator(CUDALangSpec()), CUDACompiler(c), CUDAExecutor(c))

def detect(b):
    match b:
        case "CLANG": return ClangBackend()
        case "CUDA": return CUDABackend()
        case "PYTHON": return PythonBackend()
        case _: raise RuntimeError(f"unknown WHALE_BACKEND: {b}")

bcknd = detect(os.environ.get("WHALE_BACKEND", "PYTHON"))

def to_cpu(buff):
    return bcknd.executor.memcpy_dtoh(buff.dev.ptr, buff.length, buff.dtype.ctype())

def free(ptr):
    bcknd.executor.free(ptr)

def reset(b):
    global bcknd
    bcknd = detect(b)
