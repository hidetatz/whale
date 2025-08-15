from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto

import backend
import cuda
import device
import kernel
from tensor_op import TensorOp, TensorOpCode

dbg = os.getenv("WHALE_DEBUG", "") != ""


class Instruction:
    def __str__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if k != "instid")
        return f"<{self.__class__.__name__}({params})>"


@dataclass
class AllocateDeviceMemory(Instruction):
    t: TensorOp


@dataclass
class CopyBufferPythonToDevice(Instruction):
    t: TensorOp


@dataclass
class CopyBufferDeviceToPython(Instruction):
    t: TensorOp


@dataclass
class CopyDevicePointer(Instruction):
    src: TensorOp
    dst: TensorOp


@dataclass
class InvokeUnaryKernel(Instruction):
    kern_name: str
    dst: TensorOp
    src: TensorOp


@dataclass
class InvokeBinaryKernel(Instruction):
    kern_name: str
    dst: TensorOp
    srcl: TensorOp
    srcr: TensorOp


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
            self.kernel_generator = cuda.CodeGenerator()
            self.kernel_manager = cuda.KernelManager()

        self.initialized = True

    @classmethod
    def materialize(cls, op: TensorOp) -> None:
        self = cls()  # get materializer singleton instance

        if dbg:
            print("=== materialization start ===")
            print("=== backend")
            print(self.backend)

            print("=== tensor graph")
            print(op)

        ops = self.linearize(op)

        if dbg:
            print("=== linearized tensor ops")
            print("\n".join([f"{op.str_oneline()}" for op in ops]))

        # generate and load kernels
        kerns = self.generate_kernels(ops)

        if dbg:
            print("=== kernels")
            print("\n\n".join([f"{kern.src}" for kern in kerns]))

        self.kernel_manager.load(kerns)
        insts = self.generate_instructions(ops)
        if dbg:
            print("=== instructions")
            print("\n".join([f"{inst}" for inst in insts]))

        self.execute(insts)

        if dbg:
            print("=== materialized tensor")
            print(op.str_oneline())
            print("=== materialization successfully finished ===")

    def linearize(self, op: TensorOp):
        ops = []
        seen = []

        def dfs(_op: TensorOp):
            if _op in seen:
                return

            seen.append(_op)
            ops.append(_op)

            for i in _op.inputs:
                dfs(i)

        dfs(op)
        ops.reverse()
        return ops

    def generate_kernels(self, ops: list[TensorOp]) -> list[kernel.Kernel]:
        code_map: dict[TensorOpCode, kernel.OpCode] = {
            TensorOpCode.RECIP: kernel.OpCode.RECIP,
            TensorOpCode.ADD: kernel.OpCode.ADD,
            TensorOpCode.MUL: kernel.OpCode.MUL,
            TensorOpCode.POW: kernel.OpCode.POW,
        }
        kerns: list[kernel.Kernel] = []
        for op in ops:
            if op.code.is_unary_op():
                name, src = self.kernel_generator.generate_unary_kernel(code_map[op.code], op.ndim)

            elif op.code.is_binary_op():
                name, src = self.kernel_generator.generate_binary_kernel(code_map[op.code], op.ndim)

            else:
                continue

            kerns.append(kernel.Kernel(name, src, None))

        return kerns

    def generate_instructions(self, ops: list[TensorOp]) -> list[Instruction]:
        insts: list[Instruction] = []
        for op in ops:
            if op.code.is_buffer_op():
                insts.append(AllocateDeviceMemory(op))
                insts.append(CopyBufferPythonToDevice(op))

            elif op.code.is_unary_op():
                insts.append(AllocateDeviceMemory(op))
                insts.append(InvokeUnaryKernel(kernel.to_kern_name(op.code, op.ndim), op, op.inputs[0]))

            elif op.code.is_binary_op():
                insts.append(AllocateDeviceMemory(op))
                insts.append(InvokeBinaryKernel(kernel.to_kern_name(op.code, op.ndim), op, op.inputs[0], op.inputs[1]))

        return insts

    def execute(self, insts: list[Instruction]) -> None:
        for inst in insts:
            # if type(inst) is AllocateDeviceMemory:
            if type(inst) is AllocateDeviceMemory:
                inst.t.dev_buffer = inst.t.dev.allocate(inst.t.size)

            elif type(inst) is CopyBufferPythonToDevice:
                inst.t.dev.copy_to_device(inst.t.cpu_buffer, inst.t.dev_buffer)

            elif type(inst) is CopyBufferDeviceToPython:
                inst.t.dev.copy_to_device(inst.t.dev_buffer, inst.t.cpu_buffer)

            elif type(inst) is CopyDevicePointer:
                pass

            elif type(inst) is InvokeUnaryKernel:
                params = (inst.src.offset, inst.dst.offset, *inst.src.strides, *inst.dst.strides, inst.src.dev_buffer, inst.dst.dev_buffer)
                self.kernel_manager.invoke(inst.kern_name, 1, inst.dst.shape, params)

            elif type(inst) is InvokeBinaryKernel:
                params = (
                    inst.srcl.offset,
                    inst.srcr.offset,
                    inst.dst.offset,
                    *inst.srcl.strides,
                    *inst.srcr.strides,
                    *inst.dst.strides,
                    inst.srcl.dev_buffer,
                    inst.srcr.dev_buffer,
                    inst.dst.dev_buffer,
                )
                self.kernel_manager.invoke(inst.kern_name, 1, inst.dst.shape, params)
