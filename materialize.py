import hashlib
import os

import backend
import algo
import kernel
import sched
from debug import DEBUG, dbg, dbg_tree, dbg_schedule, dbg_kernel

class Materializer:
    def __init__(self):
        self.kern_cache = kernel.KernelCache()

    def materialize(self, arr):
        if DEBUG:
            dbg("materialization start")
            dbg(f"backend: {backend.bcknd.__str__().removesuffix("Backend")}")
            dbg(f"materializing ndarray:")
            dbg_tree(arr)

        if arr.ctx.op.is_const():
            if DEBUG:
                dbg("materialization did not happend as it's constant")
            return

        if arr.ctx.op.is_view():
            if DEBUG:
                dbg("as it's view op, materializing the source ndarray...")
            materialize(arr.ctx.inputs[0])
            return

        funcs = algo.convert(arr)

        scheds = sched.schedule(funcs, isinstance(backend.bcknd, backend.GPUBackend))

        if DEBUG:
            dbg(f"{len(funcs)} funcs and schedules:")
            for i, (f, s) in enumerate(zip(funcs, scheds)):
                dbg(f"{i+1}:")
                dbg_tree(f)
                dbg_schedule(s)

        for i, (func, schedule) in enumerate(zip(funcs, scheds)):
            bufs, fncs = func.args()
            params = [func.out_buffer] + [buf.node.buffer for buf in bufs] + [f.func.out_buffer for f in fncs]

            cache_key = self.func_cache_key(func, schedule)
            # if the same kernel implementation (same func and sched) is already compiled, use it
            if self.kern_cache.has(cache_key):
                self.kern_cache.hit(cache_key)
                if DEBUG:
                    dbg(f"kernel cache hit")
                backend.bcknd.invoke_kernel(schedule, self.kern_cache.get(cache_key), params)
                continue

            kern_name = f"kern_{cache_key[:16]}"
            code = backend.bcknd.codegen(kern_name, func, schedule, bufs+fncs)
            if DEBUG:
                dbg(f"kernel codegen:")
                dbg(f"{i+1}:")
                dbg_kernel(code)

            kern = backend.bcknd.compile(kern_name, code)
            self.kern_cache.save(cache_key, kern)

            backend.bcknd.invoke_kernel(schedule, kern, params)

            if DEBUG:
                dbg(f"kernel invocation count: {len(backend.bcknd.kern_invoke_hist)}")

    def func_cache_key(self, func, schedule):
        def _expr_key(expr, buf_ids):
            match expr:
                case algo.BinaryExpr():
                    return ("Bin", expr.op.name, _expr_key(expr.l_expr, buf_ids), _expr_key(expr.r_expr, buf_ids))
                case algo.UnaryExpr():
                    return ("Un", expr.op.name, _expr_key(expr.expr, buf_ids))
                case algo.ReduceExpr():
                    return ("Red", expr.op.name, tuple((v.name, v.extent) for v in expr.reduced), _expr_key(expr.expr, buf_ids))
                case algo.IndexExpr():
                    return ("Idx", expr.loopvar.name, expr.loopvar.extent)
                case algo.ConstExpr():
                    return ("Const", expr.val)
                case algo.FuncExpr():
                    return ("Fnc", _expr_key(expr.func.expr, buf_ids), tuple(_expr_key(i, buf_ids) for i in expr.indices))
                case algo.BufferExpr():
                    nid = id(expr.node)
                    if nid not in buf_ids: buf_ids[nid] = len(buf_ids)
                    indices = tuple(_expr_key(i, buf_ids) for i in expr.indices)
                    return ("Buf", buf_ids[nid], expr.node.shape, expr.node.strides, expr.node.offset, str(expr.node.dtype), indices)
        
        def _sched_key(schedule):
            def exec_key(e):
                match e:
                    case sched.GPUBlock(): return ("GPUBlock", e.index)
                    case sched.GPUThread(): return ("GPUThread", e.index)
                    case sched.Unroll(): return ("Unroll", e.factor)
                    case _: return (type(e).__name__,)
            loops = tuple((ls.lv.name, ls.lv.extent, ls.kind.name, exec_key(ls.exec)) for ls in schedule.loops)
            splits = tuple((sp.orig.name, sp.factor, sp.tail_guard_required) for sp in schedule.splits)
            return (loops, splits)
        
        buf_ids = {}
        key = (
            tuple((lv.name, lv.extent) for lv in func.out_loops),
            func.out_shape,
            str(func.out_dtype),
            _expr_key(func.expr, buf_ids),
            _sched_key(schedule),
        )
        return hashlib.md5(str(key).encode()).hexdigest()

materializer = Materializer()

def materialize(arr):
    materializer.materialize(arr)

def reset():
    global materializer
    materializer = Materializer()
