import os

import debug
import backend
import exprir
import kernel
import sched
from debug import DEBUG

class Materializer:
    def __init__(self):
        self.kern_cache = kernel.KernelCache()

    def materialize(self, arr):
        if DEBUG:
            print("=== materialization start ===")
            print(f"=== backend: {backend.bcknd}")

        # if DEBUG: debug.debug_ndarray(arr)

        funcs = exprir.convert(arr)

        # if DEBUG: debug.debug_funcs(funcs)

        scheds = sched.schedule(funcs, isinstance(backend.bcknd, backend.GPUBackend))

        # if DEBUG: debug.debug_scheds(scheds)

        for func, schedule in zip(funcs, scheds):
            bufs, fncs = func.inputs()
            params = [func.out_buffer] + [buf.node.buffer for buf in bufs] + [f.func.out_buffer for f in fncs]

            # todo: implement cache
            # if self.kern_cache.has(func):
            #     self.bcknd.invoke_kernel(schedule, self.kern_cache.get(func), params)

            kern_name, code = backend.bcknd.codegen(func, schedule, bufs+fncs)

            # if DEBUG: print(code)

            kern = backend.bcknd.compile(kern_name, code)
            self.kern_cache.save(kern_name, kern)

            backend.bcknd.invoke_kernel(schedule, kern, params)

materializer = Materializer()

def materialize(arr):
    materializer.materialize(arr)

def reset():
    global materializer
    materializer.materialize(arr)

