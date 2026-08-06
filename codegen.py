from functools import reduce

import algo
import dtype
import sched
import util
from ops import Ops

class CodeGenerator:
    def __init__(self, langspec):
        self.langspec = langspec
        self.buff = []

    def w(self, line): self.buff.append(line)
    def render(self): return "\n".join(self.buff)

    def codegen(self, func, schedule, inputs):
        pass

class HighLevelLangCodeGenerator(CodeGenerator):
    def __init__(self, langspec):
        super().__init__(langspec)
        self.indent_level = 0
        self.tmpvar_idx = 0

    def nest(self): self.indent_level += 1
    def unnest(self): self.indent_level -= 1
    def writeln(self, ln): self.buff.append(f"{self.langspec.indent_str() * self.indent_level}{ln}")
    def write(self, code):
        if isinstance(code, str): self.writeln(code)
        elif isinstance(code, list):
            for l in code: self.writeln(l)
        else: raise RuntimeError(f"cannot handle in codegen: {code}")
    def tmpvar(self):
        n = f"tmp{self.tmpvar_idx}"
        self.tmpvar_idx += 1
        return n

    def arr_idx_calc_expr(self, shape, names):
        if not shape: return "0"
        return reduce(self.langspec.add, [self.langspec.mul(name, st) for name, st in zip(names, util.strides_from_shape(shape))])

    def codegen(self, kern_name, func, schedule, inputs):
        self.buff = []
        l = self.langspec

        for lib in l.default_library(): self.write(l.import_lib(lib))

        args = {f"{self.argname(inp)}_{i}": inp for i, inp in enumerate(inputs)}

        arg_names = ["out"] + list(args.keys())
        arg_types = [func.out_dtype] + [expr.node.dtype if isinstance(expr, algo.BufferExpr) else expr.func.out_dtype for expr in args.values()]
        self.write(l.kern_start(kern_name, arg_names, arg_types))
        self.nest()

        loop_finish_fn = self.render_loop(schedule)

        # extract original loop from split outer and inner
        for sp in schedule.splits:
            self.write(l.init(dtype.int64, sp.orig.name, l.add(l.mul(sp.o.name, sp.factor), sp.i.name)))

        for sp in schedule.splits:
            if sp.tail_guard_required:
                self.write(l.if_start(l.greater_than(sp.orig.extent, sp.orig.name)))
                self.nest()
                self.write(l.return_function())
                self.unnest()
                self.write(l.if_end())

        result = self.render_expr(func.expr, args, func.out_dtype)
        idx = self.arr_idx_calc_expr(func.out_shape, [lv.name for lv in func.out_loops])
        self.write(l.assign(l.index("out", idx), result))

        loop_finish_fn()

        self.unnest()
        self.write(l.kern_end())

        return self.render()

    def argname(self, arg):
        match arg:
            case algo.BufferExpr(): return "buf"
            case algo.FuncExpr(): return "fnc"
            case _: raise RuntimeError(f"unexpected arg type: {type(arg)}")

    def render_loop(self, schedule):
        loopcnt = 0
        for l in schedule.loops:
            if l.kind != sched.LoopKind.Spatial: continue
            looped = True
            match l.exec:
                case sched.Sequential():
                    lp = self.langspec.sequential_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Parallel():
                    lp = self.langspec.parallel_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Vectorize():
                    lp = self.langspec.vectorized_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Unroll():
                    lp = self.langspec.unrolled_loop_start(l.lv.name, 0, l.lv.extent, 1, l.exec.factor)
                case sched.GPUBlock():
                    lp = self.langspec.gpu_block_index(l.lv.name, 0, l.lv.extent, 1, l.exec.index)
                    looped = False
                case sched.GPUThread():
                    lp = self.langspec.gpu_thread_index(l.lv.name, 0, l.lv.extent, 1, l.exec.index)
                    looped = False

            self.write(lp)

            if looped:
                loopcnt += 1
                self.nest()

        def loop_finish():
            for i in range(loopcnt):
                self.unnest()
                self.write(self.langspec.loop_end())

        return loop_finish

    def render_expr(self, expr, args, dt):
        match expr:
            case algo.UnaryExpr(): return self.render_unary(expr, args, dt)
            case algo.BinaryExpr(): return self.render_binary(expr, args, dt)
            case algo.ReduceExpr(): return self.render_reduce(expr, args, dt)
            case algo.BufferExpr(): return self.render_buffer(expr, args, dt)
            case algo.FuncExpr(): return self.render_func(expr, args, dt)
            case _: raise RuntimeError(f"unexpected expr type: {type(expr)}")

    def render_unary(self, expr, args, dt):
        l = self.langspec

        if expr.op == Ops.Neg: f = l.neg
        elif expr.op == Ops.Sin: f = l.sin
        elif expr.op == Ops.Cos: f = l.cos
        elif expr.op == Ops.Exp: f = l.exp
        elif expr.op == Ops.Log: f = l.log
        elif expr.op == Ops.Sqrt: f = l.sqrt
        else: raise RuntimeError(f"unknown unary op: {expr.op}")

        result = self.render_expr(expr.expr, args, dt)
        tmpvar = self.tmpvar()
        self.write(l.init(dt, tmpvar, f(result)))
        return tmpvar

    def render_binary(self, expr, args, dt):
        l = self.langspec

        if expr.op == Ops.Add: f = l.add
        elif expr.op == Ops.Sub: f = l.sub
        elif expr.op == Ops.Mul: f = l.mul
        elif expr.op == Ops.Truediv: f = l.truediv
        elif expr.op == Ops.Pow: f = l.pow
        else: raise RuntimeError(f"unknown binary op: {expr.op}")

        left, right = self.render_expr(expr.l_expr, args, dt), self.render_expr(expr.r_expr, args, dt)
        tmpvar = self.tmpvar()
        self.write(l.init(dt, tmpvar, f(left, right)))
        return tmpvar

    def render_reduce(self, expr, args, dt):
        l = self.langspec

        acc = "acc"
        self.write(l.init(dt, acc, "0"))

        for idx in expr.reduced:
            self.write(l.sequential_loop_start(idx.name, 0, idx.extent, 1)) # for now reduce loop is not scheduled
            self.nest()

        result = self.render_expr(expr.expr, args, dt)

        if expr.op == Ops.Sum: f = l.add
        else: raise RuntimeError(f"unknown reduce op: {expr.op}")

        self.write(l.assign(acc, f(acc, result)))

        for idx in expr.reduced:
            self.unnest()
            self.write(l.loop_end())

        return acc

    def render_buffer(self, expr, args, dt):
        # get buffer arg name from BufferExpr.node
        buf = ""
        for name, e in args.items():
            if isinstance(e, algo.BufferExpr) and e.node is expr.node:
                buf = name
                break
        assert buf != "", "expected buffer is not found in args"

        names = [idx.loopvar.name if isinstance(idx, algo.IndexExpr) else str(idx.val) for idx in expr.indices]
        l = self.langspec
        terms = [l.mul(name, str(st)) for name, st in zip(names, expr.node.strides) if st != 0]
        flat = reduce(l.add, [str(expr.node.offset)] + terms) if terms else str(expr.node.offset)
        return l.index(buf, flat)

    def render_func(self, expr, args, dt):
        fnc = ""
        for name, e in args.items():
            if isinstance(e, algo.FuncExpr) and e.func is expr.func:
                fnc = name
                break
        assert fnc != "", "expected func result is not found in args"
        idx = self.arr_idx_calc_expr(expr.func.out_shape, [idx.loopvar.name if isinstance(idx, algo.IndexExpr) else str(idx.val) for idx in expr.indices])
        return self.langspec.index(fnc, idx)
