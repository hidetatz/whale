from dataclasses import dataclass
from enum import Enum, auto
import os

import backend
import cuda
import device

# from tensor import Tensor
from tensor_op import TensorOp

dbg = os.getenv("WHALE_DEBUG", "") != ""


class Materializer:
    # materializer is singleton for caching some info
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return

        self.backend = backend.Backend.detect()
        if self.backend == backend.Backend.CUDA:
            self.renderer = cuda.Renderer()
            self.allocator = cuda.Allocator()
            self.kernel_manager = cuda.KernelManager()

        self.initialized = True

    @classmethod
    def materialize(cls, t):
        self = cls()  # get materializer singleton instance

        if dbg:
            print("=== materialization start ===")
            print("=== backend")
            print(self.backend)

            print("=== tensor graph")
            print(t.str_as_graph())

        tensors = self.linearize(t)

        if dbg:
            print("=== linearized tensors")
            print("\n".join([f"{t.str_as_oneline()}" for t in tensors]))

        # generate and load kernels
        kernel_srcs = self.generate_kernels(tensors)

        if dbg:
            print("=== kernels")
            print("\n\n".join([f"{src}" for name, src in kernel_srcs.items()]))

        self.kernel_manager.load([src for _, src in kernel_srcs.items()])
        self.execute(tensors, self.allocator, self.kernel_manager)

        if dbg:
            print("=== materialization successfully finished ===")

    def linearize(self, t):
        tensors = []
        seen = []

        def dfs(_t):
            if _t in seen:
                return

            seen.append(_t)
            tensors.append(_t)

            for i in _t.inputs:
                dfs(i)

        dfs(t)
        tensors.reverse()
        return tensors

    def generate_kernels(self, tensors):
        kerns = {}
        for t in tensors:
            if t.op == TensorOp.ADD:
                kern_src, kern_name = self.renderer.render_kern_add(t.shape, t.strides, t.offset)
            elif t.op == TensorOp.MUL:
                kern_src, kern_name = self.renderer.render_kern_mul(t.shape, t.strides, t.offset)
            elif t.op == TensorOp.RECIP:
                kern_src, kern_name = self.renderer.render_kern_recip(t.shape, t.strides, t.offset)
            elif t.op == TensorOp.POW:
                kern_src, kern_name = self.renderer.render_kern_pow(t.shape, t.strides, t.offset)
            else:
                continue

            kerns[kern_name] = device.KernelSrc(kern_src, kern_name)

        return kerns

    def execute(self, tensors: list[any], allocator: device.Allocator, kernel_manager: device.KernelManager):
        for i, t in enumerate(tensors):
            if t.op == TensorOp.ADD:
                l = t.inputs[0]
                r = t.inputs[1]
                if l.base.device_buffer is None:
                    l.base.device_buffer = allocator.allocate(l.base.size)
                    allocator.copy_to_device(l.base.python_buffer, l.base.device_buffer)

                if r.base.device_buffer is None:
                    r.base.device_buffer = allocator.allocate(r.base.size)
                    allocator.copy_to_device(r.base.python_buffer, r.base.device_buffer)

                t.device_buffer = allocator.allocate(t.size)
                kernname = f"add_{'_'.join(map(str, t.shape))}"
                strides = t.strides if t.strides else (0,)
                kernel_manager.invoke(
                    kernname, 1, t.shape if t.shape else 1, [t.offset, *strides, l.base.device_buffer, r.base.device_buffer, t.device_buffer]
                )

            elif t.op == TensorOp.MUL:
                l = t.inputs[0]
                r = t.inputs[1]
                if l.base.device_buffer is None:
                    l.base.device_buffer = allocator.allocate(l.base.size)
                    allocator.copy_to_device(l.base.python_buffer, l.base.device_buffer)

                if r.base.device_buffer is None:
                    r.base.device_buffer = allocator.allocate(r.base.size)
                    allocator.copy_to_device(r.base.python_buffer, r.base.device_buffer)

                t.device_buffer = allocator.allocate(t.size)
                kernname = f"mul_{'_'.join(map(str, t.shape))}"
                strides = t.strides if t.strides else (0,)
                kernel_manager.invoke(
                    kernname, 1, t.shape if t.shape else 1, [t.offset, *strides, l.base.device_buffer, r.base.device_buffer, t.device_buffer]
                )

            elif t.op == TensorOp.RECIP:
                r = t.inputs[0]
                if r.base.device_buffer is None:
                    r.base.device_buffer = allocator.allocate(r.base.size)
                    allocator.copy_to_device(r.base.python_buffer, r.base.device_buffer)

                t.device_buffer = allocator.allocate(t.size)
                kernname = f"recip_{'_'.join(map(str, t.shape))}"
                strides = t.strides if t.strides else (0,)
                kernel_manager.invoke(kernname, 1, t.shape if t.shape else 1, [t.offset, *strides, r.base.device_buffer, t.device_buffer])

            elif t.op == TensorOp.POW:
                l = t.inputs[0]
                r = t.inputs[1]
                if l.base.device_buffer is None:
                    l.base.device_buffer = allocator.allocate(l.base.size)
                    allocator.copy_to_device(l.base.python_buffer, l.base.device_buffer)

                if r.base.device_buffer is None:
                    r.base.device_buffer = allocator.allocate(r.base.size)
                    allocator.copy_to_device(r.base.python_buffer, r.base.device_buffer)

                t.device_buffer = allocator.allocate(t.size)
                kernname = f"pow_{'_'.join(map(str, t.shape))}"
                strides = t.strides if t.strides else (0,)
                kernel_manager.invoke(
                    kernname, 1, t.shape if t.shape else 1, [t.offset, *strides, l.base.device_buffer, r.base.device_buffer, t.device_buffer]
                )

            if i == len(tensors) - 1:
                allocator.copy_from_device(t.device_buffer, t.python_buffer)
