from abc import ABC, abstractmethod

import dtype

class LangSpec(ABC): pass

class HighLevelLangSpec(LangSpec):
    # library and import
    @abstractmethod
    def import_lib(self, lib): ...
    @abstractmethod
    def default_library(self): ...

    # language
    @abstractmethod
    def indent_str(self): ...

    # kernel
    @abstractmethod
    def kern_start(self, name, arg_names, arg_types): ...
    @abstractmethod
    def kern_end(self): ...

    # branch and loop
    @abstractmethod
    def sequential_loop_start(self, index, start, end, step): ...
    @abstractmethod
    def parallel_loop_start(self, index, start, end, step): ...
    @abstractmethod
    def vectorized_loop_start(self, index, start, end, step): ...
    @abstractmethod
    def unrolled_loop_start(self, index, start, end, step, factor): ...
    @abstractmethod
    def gpu_block_index(self, index, start, end, step, idx): ...
    @abstractmethod
    def gpu_thread_index(self, index, start, end, step, idx): ...
    @abstractmethod
    def loop_end(self): ...
    @abstractmethod
    def if_start(self, cond): ...
    @abstractmethod
    def if_end(self): ...

    # function
    @abstractmethod
    def return_function(self): ...

    # access
    @abstractmethod
    def index(self, a, idx): ...
    @abstractmethod
    def init(self, dt, l, r): ...
    @abstractmethod
    def assign(self, l, r): ...

    # unary
    @abstractmethod
    def neg(self, a): ...
    @abstractmethod
    @abstractmethod
    def sin(self, a): ...
    @abstractmethod
    def cos(self, a): ...
    @abstractmethod
    def exp(self, a): ...
    @abstractmethod
    def log(self, a): ...
    @abstractmethod
    def sqrt(self, a): ...

    # binary
    @abstractmethod
    def add(self, l, r): ...
    @abstractmethod
    def sub(self, l, r): ...
    @abstractmethod
    def mul(self, l, r): ...
    @abstractmethod
    def truediv(self, l, r): ...
    @abstractmethod
    def floordiv(self, l, r): ...
    @abstractmethod
    def mod(self, l, r): ...
    @abstractmethod
    def pow(self, l, r): ...
    @abstractmethod
    def _and(self, l, r): ...
    @abstractmethod
    def _or(self, l, r): ...
    @abstractmethod
    def eq(self, l, r): ...
    @abstractmethod
    def gt(self, l, r): ...
    @abstractmethod
    def ge(self, l, r): ...
    @abstractmethod
    def lt(self, l, r): ...
    @abstractmethod
    def le(self, l, r): ...

    # ternary
    @abstractmethod
    def where(self, e1, e2, e3): ...

class CCompatibleLangSpec(HighLevelLangSpec):
    def typename(self, dt):
        if dt == dtype.int32: return "int32_t"
        elif dt == dtype.int64: return "int64_t"
        elif dt == dtype.float32: return "float"
        elif dt == dtype.float64: return "double"
        else: raise RuntimeError(f"unknown dtype: {dt}")
    def import_lib(self, lib): return f"#include <{lib}>"
    def default_library(self): return ["stdint.h", "math.h"]
    def indent_str(self): return "    "

    def kern_end(self): return "}"
    def sequential_loop_start(self, index, start, end, step): return f"for (int {index} = {start}; {index} < {end}; {index} += {step}) {{"
    def loop_end(self): return "}"
    def if_start(self, cond): return f"if ({cond}) {{"
    def if_end(self): return "}"
    def return_function(self): return "return;"
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
    def floordiv(self, l, r): return f"{l} / {r}"
    def mod(self, l, r): return f"{l} % {r}"
    def pow(self, l, r): return f"pow({l}, {r})"
    def _and(self, l, r): return f"({l} && {r})"
    def _or(self, l, r): return f"({l} || {r})"
    def eq(self, l, r): return f"({l} == {r})"
    def gt(self, l, r): return f"({l} > {r})"
    def ge(self, l, r): return f"({l} >= {r})"
    def lt(self, l, r): return f"({l} < {r})"
    def le(self, l, r): return f"({l} <= {r})"
    def where(self, e1, e2, e3): return f"{e1} ? {e2} : {e3}"

    @abstractmethod
    def kern_start(self, name, arg_names, arg_types): ...
    @abstractmethod
    def parallel_loop_start(self, index, start, end, step): ...
    @abstractmethod
    def vectorized_loop_start(self, index, start, end, step): ...
    @abstractmethod
    def unrolled_loop_start(self, index, start, end, step, factor): ...
    @abstractmethod
    def gpu_block_index(self, index, start, end, step, idx): ...
    @abstractmethod
    def gpu_thread_index(self, index, start, end, step, idx): ...

