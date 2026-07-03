class Python:
    def __init__(self):
        self.kerns = {}

    @classmethod
    def is_gpu(cls): return False

    # lang settings
    def import_lib(self, lib): return f"import {lib}"
    def default_library(self): return ["math"]
    def indent_str(self): return "    "
    def kern_start(self, name, arg_names, arg_types): return f"def {name}({', '.join(arg_names)}):"
    def kern_end(self): return ""
    def loop_start(self, index, start, end, step): return f"for {index} in range({start}, {end}, {step}):"
    def loop_end(self): return ""
    def index(self, a, idx): return f"{a}[{idx}]"
    def init(self, dt, l, r): return f"{l} = {r}"
    def assign(self, l, r): return f"{l} = {r}"
    def neg(self, a): return f"-({a})"
    def pow(self, a): return f"pow({a}, 2)"
    def sin(self, a): return f"math.sin({a})"
    def cos(self, a): return f"math.cos({a})"
    def exp(self, a): return f"math.exp({a})"
    def log(self, a): return f"math.log({a})"
    def sqrt(self, a): return f"math.sqrt({a})"
    def add(self, l, r): return f"{l} + {r}"
    def sub(self, l, r): return f"{l} - {r}"
    def mul(self, l, r): return f"{l} * {r}"
    def truediv(self, l, r): return f"{l} / {r}"

    # exec settings
    def compile(self, name, code): exec(code, self.kerns)
    def execute(self, name, param_buffs): self.kerns[name](*[buff.cpu.val for buff in param_buffs])
