import math
from functools import reduce

import buffer
import dtype
import exprir
import sched
from ops import Ops
from backend_python import Python

def strides_from_shape(shape):
    return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])

class CodeGenerator:
    def __init__(self, lang):
        self.lang = lang
        self.buff = []

    def w(self, line): self.buff.append(line)
    def render(self): return "\n".join(self.buff)

    def codegen(self, func, schedule):
        pass

class CLikeCodeGenerator(CodeGenerator):
    def __init__(self, lang):
        self.indent_level = 0
        self.tmpvar_idx = 0
        super().__init__(lang)

    def nest(self): self.indent_level += 1
    def unnest(self): self.indent_level -= 1
    def write(self, code): self.buff.append(f"{self.lang.indent_str() * self.indent_level}{code}{';' if self.lang.require_semicolon() else ''}")
    def tmpvar(self):
        n = f"tmp{self.tmpvar_idx}"
        self.tmpvar_idx += 1
        return n

    def arr_idx_calc_expr(self, shape, names):
        if not shape: return "0"
        return reduce(self.lang.add, [self.lang.mul(name, st) for name, st in zip(names, strides_from_shape(shape))])

    def codegen(self, func, schedule):
        l = self.lang

        bufs, fncs = func.inputs()
        kern_name = f"kern_{id(func)}"
        args = {f"{self.argname(buf)}_{i}": buf for i, buf in enumerate(bufs)} | {f"{self.argname(fnc)}_{i}": fnc for i, fnc in enumerate(fncs)}
        arg_names = ["out"] + list(args.keys())

        self.write(l.kern_start(kern_name, arg_names))
        self.nest()

        for idx in func.out_indices:
            self.write(l.loop_start(idx.name, 0, idx.extent, 1))
            self.nest()

        result = self.render_expr(func.expr, args)
        idx = self.arr_idx_calc_expr(func.out_shape, [idx.name for idx in func.out_indices])
        self.write(l.assign(l.index("out", idx), result))

        for idx in func.out_indices:
            self.unnest()
            self.write(l.loop_end())

        self.unnest()
        self.write(l.kern_end())

        return kern_name, self.render()

    def argname(self, arg):
        match arg:
            case exprir.BufferExpr(): return "buf"
            case exprir.FuncExpr(): return "fnc"
            case _: raise RuntimeError(f"unexpected arg type: {type(arg)}")

    def render_expr(self, expr, args):
        match expr:
            case exprir.UnaryExpr(): return self.render_unary(expr, args)
            case exprir.BinaryExpr(): return self.render_binary(expr, args)
            case exprir.ReduceExpr(): return self.render_reduce(expr, args)
            case exprir.BufferExpr(): return self.render_buffer(expr, args)
            case _: raise RuntimeError(f"unexpected expr type: {type(expr)}")

    def render_unary(self, expr, args):
        l = self.lang

        if expr.op == Ops.Neg: f = l.neg
        elif expr.op == Ops.Pow: f = l.pow
        elif expr.op == Ops.Sin: f = l.sin
        elif expr.op == Ops.Cos: f = l.cos
        elif expr.op == Ops.Exp: f = l.exp
        elif expr.op == Ops.Log: f = l.log
        elif expr.op == Ops.Sqrt: f = l.sqrt
        else: raise RuntimeError(f"unknown unary op: {expr.op}")

        result = self.render_expr(expr.operand, args)
        tmpvar = self.tmpvar()
        self.write(l.init(tmpvar, f(result)))
        return tmpvar

    def render_binary(self, expr, args):
        l = self.lang

        if expr.op == Ops.Add: f = l.add
        elif expr.op == Ops.Sub: f = l.sub
        elif expr.op == Ops.Mul: f = l.mul
        elif expr.op == Ops.Truediv: f = l.truediv
        else: raise RuntimeError(f"unknown binary op: {expr.op}")

        left, right = self.render_expr(expr.left, args), self.render_expr(expr.right, args)
        tmpvar = self.tmpvar()
        self.write(l.init(tmpvar, f(left, right)))
        return tmpvar

    def render_reduce(self, expr, args):
        l = self.lang

        acc = "acc"
        self.write(l.init(acc, "0"))

        for idx in expr.reduced:
            self.write(l.loop_start(idx.name, 0, idx.extent, 1))
            self.nest()

        result = self.render_expr(expr.operand, args)

        if expr.op == Ops.Sum: f = l.add
        else: raise RuntimeError(f"unknown reduce op: {expr.op}")

        self.write(l.assign(acc, f(acc, result)))

        for idx in expr.reduced:
            self.unnest()
            self.write(l.loop_end())

        return acc

    def render_buffer(self, expr, args):
        buf = list(args.keys())[list(args.values()).index(expr)]  # get buffer arg name from BufferExpr instance
        idx = self.arr_idx_calc_expr(expr.src.shape, [idx.idx.name for idx in expr.indices])
        return self.lang.index(buf, idx)

_backend = Python

def gpu_enabled():
    return _backend.is_gpu()

def lower_and_exec(eir, scheds):
    b = _backend()
    for func, schedule in zip(eir.funcs, scheds):
        codegenerator = CLikeCodeGenerator(b)
        kern_name, code = codegenerator.codegen(func, schedule)
        bufs, fncs = func.inputs()
        params = [func.out_buffer] + [b.src.buffer for b in bufs] + [f.src.out_buffer for f in fncs]
        b.compile(code)
        b.execute(kern_name, params)

if __name__ == "__main__":
    from ndarray import array, _const
    a = _const([2, 3], [i for i in range(6)])
    b = _const([2, 3], [1, 2, 3, 1, 2, 3])
    c = a + b
    d = c.sum(axis=[0])
    c.materialize()

    print(c.tolist())
