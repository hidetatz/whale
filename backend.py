from functools import reduce
import operator

import buffer
import dtype
import exprir
import sched
from ops import Ops

class Backend:
    def codegen(self, func, schedule):
        self.tmpvar_idx = 0
        renderer = self.get_renderer()

        bufs, fncs = func.inputs()
        kern_name = f"aaa"
        self.args = {f"{self.argname(buf)}_{i}": buf for i, buf in enumerate(bufs)} | {f"{self.argname(fnc)}_{i}": fnc for i, fnc in enumerate(fncs)}
        arg_names = ["out"] + list(self.args.keys())

        renderer.kern_start(kern_name, arg_names)

        for idx in func.out_indices:
            renderer.loop_start(idx.name, 0, idx.extent, 1)

        result_var = self.codegen_expr(func.expr, renderer)

        strides = strides_from_shape(func.out_shape)
        access = [f"{i.name} * {st}" for i, st in zip(func.out_indices, strides)]
        renderer.assign(f"out[{' + '.join(access)}]", result_var)

        for idx in func.out_indices:
            renderer.loop_end()

        renderer.kern_end()

        return kern_name, renderer.render()

    def tmp_var(self):
        n = f"tmp{self.tmpvar_idx}"
        self.tmpvar_idx += 1
        return n

    def codegen_expr(self, expr, renderer):
        if isinstance(expr, exprir.UnaryExpr): return self.codegen_unary(expr, renderer)
        elif isinstance(expr, exprir.BinaryExpr): return self.codegen_binary(expr, renderer)
        elif isinstance(expr, exprir.ReduceExpr): return self.codegen_reduce(expr, renderer)
        elif isinstance(expr, exprir.BufferExpr): return self.codegen_buffer(expr, renderer)

    def codegen_unary(self, expr, renderer):
        tmpvar = self.tmp_var()
        renderer.init_var(tmpvar, 0, dtype.float64)
        if expr.op == Ops.Neg:
            renderer.ineg(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Pow:
            renderer.ipow(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Sin:
            renderer.isin(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Cos:
            renderer.icos(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Exp:
            renderer.iexp(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Log:
            renderer.ilog(tmpvar, self.codegen_expr(expr.operand, renderer))
        elif expr.op == Ops.Sqrt:
            renderer.isqrt(tmpvar, self.codegen_expr(expr.operand, renderer))
        return tmpvar

    def codegen_binary(self, expr, renderer):
        tmpvar = self.tmp_var()
        renderer.init_var(tmpvar, 0, dtype.float64)
        if expr.op == Ops.Add:
            renderer.assign_add(tmpvar, self.codegen_expr(expr.left, renderer), self.codegen_expr(expr.right, renderer))
        elif expr.op == Ops.Sub:
            renderer.assign_sub(tmpvar, self.codegen_expr(expr.left, renderer), self.codegen_expr(expr.right, renderer))
        elif expr.op == Ops.Mul:
            renderer.assign_mul(tmpvar, self.codegen_expr(expr.left, renderer), self.codegen_expr(expr.right, renderer))
        elif expr.op == Ops.Truediv:
            renderer.assign_truediv(tmpvar, self.codegen_expr(expr.left, renderer), self.codegen_expr(expr.right, renderer))
        return tmpvar

    def codegen_reduce(self, expr, renderer):
        renderer.init_var("acc", 0, dtype.float64)

        for idx in expr.reduced:
            renderer.loop_start(idx.name, 0, idx.extent, 1)

        if expr.op == Ops.Sum:
            renderer.iadd("acc", self.codegen_expr(expr.operand, renderer))

        for idx in expr.reduced:
            renderer.loop_end()

        return "acc"

    def codegen_buffer(self, expr, renderer):
        buf = list(self.args.keys())[list(self.args.values()).index(expr)]
        strides = strides_from_shape(expr.src.shape)
        access = [f"{i.idx.name} * {st}" for i, st in zip(expr.indices, strides)]
        return f"{buf}[{" + ".join(access)}]"

    def argname(self, arg):
        if type(arg) is exprir.BufferExpr: return "buf"
        if type(arg) is exprir.FuncExpr: return "fnc"
        raise RuntimeError(f"unexpected arg type: {type(arg)}")

def prod(iterable): return reduce(operator.mul, iterable, 1)
def strides_from_shape(shape): return [prod(shape[i+1:]) for i in range(len(shape))]

class Renderer:
    def __init__(self):
        self.buff = []
        self.level = 0

    def write(self, code): self.buff.append(f"{self.indent()}{code}")
    def indent(self): return self.indent_str() * self.level
    def render(self): return "\n".join(self.buff)

class PythonRenderer(Renderer):
    def indent_str(self): return "    "

    def kern_start(self, name, arg_names):
        self.write(f"def {name}({', '.join(arg_names)}):")
        self.level += 1

    def kern_end(self):
        self.level -= 1

    def loop_start(self, index, start, end, step):
        self.write(f"for {index} in range({start}, {end}, {step}):")
        self.level += 1

    def loop_end(self):
        self.level -= 1

    def init_var(self, name, value, dtype):
        self.write(f"{name} = {value}")

    def assign(self, l, r):
        return self.write(f"{l} = {r}")

    def iadd(self, l, r): self.write(f"{l} += {r}")
    def ineg(self, l, r): self.write(f"{l} = -({r})")
    def ipow(self, l, r): self.write(f"{l} = pow({r}, 2)")
    def isin(self, l, r): self.write(f"{l} = math.sin({r})")
    def icos(self, l, r): self.write(f"{l} = math.cos({r})")
    def iexp(self, l, r): self.write(f"{l} = math.exp({r})")
    def ilog(self, l, r): self.write(f"{l} = math.log({r})")
    def isqrt(self, l, r): self.write(f"{l} = math.sqrt({r})")

    def assign_add(self, left, l, r): self.write(f"{left} = {l} + {r}")
    def assign_sub(self, left, l, r): self.write(f"{left} = {l} - {r}")
    def assign_mul(self, left, l, r): self.write(f"{left} = {l} * {r}")
    def assign_truediv(self, left, l, r): self.write(f"{left} = {l} / {r}")

class PythonExecutor:
    def compile(self, code):
        self.kerns = {}
        exec(code, self.kerns)

    def execute(self, name, param_buffs):
        params = [buff.cpu.val for buff in param_buffs]
        k = self.kerns[name]
        k(*params)

class Python(Backend):
    def name(self): return "Python"
    def is_gpu(self): return False
    def get_renderer(self): return PythonRenderer()

def detect_backend():
    return Python()

_backend = detect_backend()

def gpu_enabled():
    return _backend.is_gpu()

def lower_and_exec(eir, scheds):
    e = PythonExecutor()
    for func, schedule in zip(eir.funcs, scheds):
        kern_name, code = _backend.codegen(func, schedule)
        print(func, schedule, code)

        bufs, fncs = func.inputs()
        params = [func.out_buffer] + [b.src.buffer for b in bufs] + [f.src.out_buffer for f in fncs]
        e.compile(code)
        e.execute(kern_name, params)

if __name__ == "__main__":
    from ndarray import array, _const
    # a = _const([2, 3, 4, 5], [i for i in range(120)])
    # e = a.sum(axis=[0, 2])
    # [[0, 1, 2], [3, 4, 5]]
    a = _const([2, 3], [i for i in range(6)])
    # [[1, 2, 3], [1, 2, 3]]
    b = _const([2, 3], [1, 2, 3, 1, 2, 3])
    c = a + b
    d = c.sum(axis=[0])
    # print(e.debug())
    eir = exprir.convert(d)
    # print(eir)
    scheds = sched.schedule(eir)
    lower_and_exec(eir, scheds)

    print(d.tolist())
