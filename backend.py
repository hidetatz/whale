import os
from functools import reduce

import buffer
import dtype
import exprir
import sched
import util
from ops import Ops
from buffer import DevBuff
from backend_clang import ClangC
from backend_cuda import CUDA
from backend_python import Python

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
    def writeln(self, ln): self.buff.append(f"{self.lang.indent_str() * self.indent_level}{ln}")
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
        return reduce(self.lang.add, [self.lang.mul(name, st) for name, st in zip(names, util.strides_from_shape(shape))])

    def codegen(self, func, schedule):
        l = self.lang

        for lib in l.default_library(): self.write(l.import_lib(lib))

        bufs, fncs = func.inputs()
        kern_name = f"kern_{id(func)}"
        args = {f"{self.argname(buf)}_{i}": buf for i, buf in enumerate(bufs)} | {f"{self.argname(fnc)}_{i}": fnc for i, fnc in enumerate(fncs)}

        arg_names = ["out"] + list(args.keys())
        arg_types = [func.out_dtype] + [expr.node.dtype if isinstance(expr, exprir.BufferExpr) else expr.func.out_dtype for expr in args.values()]
        self.write(l.kern_start(kern_name, arg_names, arg_types))
        self.nest()

        loop_finish_fn = self.render_loop(schedule)

        # extract original loop from split outer and inner
        for sp in schedule.splits:
            self.write(l.init(dtype.int64, sp.orig.name, l.add(l.mul(sp.o.name, sp.factor), sp.i.name)))

        result = self.render_expr(func.expr, args, func.out_dtype)
        idx = self.arr_idx_calc_expr(func.out_shape, [lv.name for lv in func.out_loops])
        self.write(l.assign(l.index("out", idx), result))

        loop_finish_fn()

        self.unnest()
        self.write(l.kern_end())

        return kern_name, self.render()

    def argname(self, arg):
        match arg:
            case exprir.BufferExpr(): return "buf"
            case exprir.FuncExpr(): return "fnc"
            case _: raise RuntimeError(f"unexpected arg type: {type(arg)}")

    def render_loop(self, schedule):
        loopcnt = 0
        for l in schedule.loops:
            if l.kind != sched.LoopKind.Spatial: continue
            looped = True
            match l.exec:
                case sched.Sequential():
                    lp = self.lang.sequential_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Parallel():
                    lp = self.lang.parallel_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Vectorize():
                    lp = self.lang.vectorized_loop_start(l.lv.name, 0, l.lv.extent, 1)
                case sched.Unroll():
                    lp = self.lang.unrolled_loop_start(l.lv.name, 0, l.lv.extent, 1, l.exec.factor)
                case sched.GPUBlock():
                    lp = self.lang.gpu_block_index(l.lv.name, 0, l.lv.extent, 1, l.exec.index)
                    looped = False
                case sched.GPUThread():
                    lp = self.lang.gpu_thread_index(l.lv.name, 0, l.lv.extent, 1, l.exec.index)
                    looped = False

            self.write(lp)

            if looped:
                loopcnt += 1
                self.nest()

        def loop_finish():
            for i in range(loopcnt):
                self.unnest()
                self.write(self.lang.loop_end())

        return loop_finish

    def render_expr(self, expr, args, dt):
        match expr:
            case exprir.UnaryExpr(): return self.render_unary(expr, args, dt)
            case exprir.BinaryExpr(): return self.render_binary(expr, args, dt)
            case exprir.ReduceExpr(): return self.render_reduce(expr, args, dt)
            case exprir.BufferExpr(): return self.render_buffer(expr, args, dt)
            case exprir.FuncExpr(): return self.render_func(expr, args, dt)
            case _: raise RuntimeError(f"unexpected expr type: {type(expr)}")

    def render_unary(self, expr, args, dt):
        l = self.lang

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
        l = self.lang

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
        l = self.lang

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
            if isinstance(e, exprir.BufferExpr) and e.node is expr.node:
                buf = name
                break
        assert buf != "", "expected buffer is not found in args"
        idx = self.arr_idx_calc_expr(expr.node.shape, [idx.loopvar.name if isinstance(idx, exprir.IndexExpr) else str(idx.val) for idx in expr.indices])
        return self.lang.index(buf, idx)

    def render_func(self, expr, args, dt):
        fnc = ""
        for name, e in args.items():
            if isinstance(e, exprir.FuncExpr) and e.func is expr.func:
                fnc = name
                break
        assert fnc != "", "expected func result is not found in args"
        idx = self.arr_idx_calc_expr(expr.func.out_shape, [idx.loopvar.name if isinstance(idx, exprir.IndexExpr) else str(idx.val) for idx in expr.indices])
        return self.lang.index(fnc, idx)

def detect(_b):
    match _b:
        case "CLANG_C": return ClangC()
        case "CUDA": return CUDA()
        case "PYTHON": return Python()
        case _: raise RuntimeError(f"unknown WHALE_BACKEND: {b}")

b = detect(os.environ.get("WHALE_BACKEND", "PYTHON"))

def set_backend(_b):
    new_b = detect(_b)
    global b
    b = new_b

def is_gpu(): return b.is_gpu()

def to_cpu(buff):
    return b.memcpy_dtoh(buff.dev.ptr, buff.length, buff.dtype.ctype())

def free(ptr):
    b.free(ptr)

def codegen_and_exec(funcs, scheds):
    for func, schedule in zip(funcs, scheds):
        codegenerator = CLikeCodeGenerator(b)
        kern_name, code = codegenerator.codegen(func, schedule)
        bufs, fncs = func.inputs()
        params = [func.out_buffer] + [buf.node.buffer for buf in bufs] + [f.func.out_buffer for f in fncs]
        b.compile(kern_name, code)

        if b.is_gpu():
            for i, p in enumerate(params):
                if p.dev.ptr is None:
                    # memalloc
                    p.dev.ptr = b.memalloc(p.length, p.dtype.ctype())

                # memcpy
                if i != 0 and p.cpu is not None: b.memcpy_htod(p.dev.ptr, p.cpu.val, p.length, p.dtype.ctype())
        else:
            for i, p in enumerate(params):
                if p.cpu is None:
                    p.cpu = buffer.CPUBuff([0] * p.length)

        b.execute(kern_name, params)
