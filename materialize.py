import os

import debug
import backend
import exprir
import kernel
import sched
from debug import DEBUG, dbg, dbg_tree, dbg_kernel

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
                dbg_tree(s)

        for i, (func, schedule) in enumerate(zip(funcs, scheds)):
            bufs, fncs = func.inputs()
            params = [func.out_buffer] + [buf.node.buffer for buf in bufs] + [f.func.out_buffer for f in fncs]

            # todo: implement cache
            # if self.kern_cache.has(func):
            #     self.bcknd.invoke_kernel(schedule, self.kern_cache.get(func), params)

            kern_name, code = backend.bcknd.codegen(func, schedule, bufs+fncs)

            if DEBUG:
                dbg(f"kernel codegen:")
                dbg(f"{i+1}:")
                dbg_kernel(code)

            kern = backend.bcknd.compile(kern_name, code)
            self.kern_cache.save(kern_name, kern)

            backend.bcknd.invoke_kernel(schedule, kern, params)

materializer = Materializer()

def materialize(arr):
    materializer.materialize(arr)

def reset():
    global materializer
    materializer.materialize(arr)

