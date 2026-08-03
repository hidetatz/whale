import os

import backend
import exprir
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

        funcs = exprir.convert(arr)

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

            kern_name = self.gen_kern_name(func, schedule)

            # if the same kernel implementation (same func and sched) is already compiled, use it
            # if self.kern_cache.has(kern_name): backend.bcknd.invoke_kernel(schedule, self.kern_cache.get(kern_name), params)

            code = backend.bcknd.codegen(kern_name, func, schedule, bufs+fncs)

            if DEBUG:
                dbg(f"kernel codegen:")
                dbg(f"{i+1}:")
                dbg_kernel(code)

            kern = backend.bcknd.compile(kern_name, code)
            self.kern_cache.save(kern_name, kern)

            backend.bcknd.invoke_kernel(schedule, kern, params)

    def gen_kern_name(self, func, schedule):
        return f"kern_{id(func)}"

materializer = Materializer()

def materialize(arr):
    materializer.materialize(arr)
