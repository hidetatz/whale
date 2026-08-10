import compiler
import executor
import langspec
from kernel import Kernel

class PythonLangSpec(langspec.HighLevelLangSpec):
    def import_lib(self, lib): return f"import {lib}"
    def default_library(self): return ["math"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"def {name}({', '.join(arg_names)}):"
    def kern_end(self): return ""
    def loop_start(self, index, start, end, step): return f"for {index} in range({start}, {end}, {step}):"
    def sequential_loop_start(self, index, start, end, step): return self.loop_start(index, start, end, step)
    def parallel_loop_start(self, index, start, end, step): return self.loop_start(index, start, end, step)
    def vectorized_loop_start(self, index, start, end, step): return self.loop_start(index, start, end, step)
    def unrolled_loop_start(self, index, start, end, step, factor):return self.loop_start(index, start, end, step)
    def gpu_block_index(self, index, start, end, step, idx): raise RuntimeError("Python backend cannot handle gpu blocks!")
    def gpu_thread_index(self, index, start, end, step, idx): raise RuntimeError("Python backend cannot handle gpu blocks!")
    def loop_end(self): return ""
    def if_start(self, cond): return f"if {cond}:"
    def if_end(self): return ""
    def return_function(self): return "return"
    def index(self, a, idx): return f"{a}[{idx}]"
    def init(self, dt, l, r): return f"{l} = {r}"
    def assign(self, l, r): return f"{l} = {r}"
    def neg(self, a): return f"-({a})"
    def sin(self, a): return f"math.sin({a})"
    def cos(self, a): return f"math.cos({a})"
    def exp(self, a): return f"math.exp({a})"
    def log(self, a): return f"math.log({a})"
    def sqrt(self, a): return f"math.sqrt({a})"
    def add(self, l, r): return f"{l} + {r}"
    def sub(self, l, r): return f"{l} - {r}"
    def mul(self, l, r): return f"{l} * {r}"
    def truediv(self, l, r): return f"{l} / {r}"
    def floordiv(self, l, r): return f"{l} // {r}"
    def mod(self, l, r): return f"{l} % {r}"
    def pow(self, l, r): return f"pow({l}, {r})"
    def _and(self, l, r): return f"{l} and {r}"
    def _or(self, l, r): return f"{l} or {r}"
    def eq(self, l, r): return f"{l} == {r}"
    def gt(self, l, r): return f"{l} > {r}"
    def ge(self, l, r): return f"{l} >= {r}"
    def lt(self, l, r): return f"{l} < {r}"
    def le(self, l, r): return f"{l} <= {r}"
    def where(self, e1, e2, e3): return f"{e2} if {e1} else {e3}"

class PythonCompiler(compiler.Compiler):
    def compile(self, name: str, code: str):
        ns = {}
        exec(code, ns)
        return Kernel(ns[name])

class PythonExecutor(executor.CPUExecutor):
    def execute(self, kern: Kernel, params: list[buffer.Buffer]):
        kern.bin(*[p.cpu.val for p in params])
