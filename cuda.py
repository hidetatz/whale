import os
import time
import typing
from ctypes import CDLL, byref, c_float, c_int, c_uint, c_void_p, cast, pointer, sizeof

import device
import kernel


class CUDA:
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")

        result = self.libcuda.cuInit(0)
        if result != 0:
            raise RuntimeError(f"cuInit failed: {result}")

        dev = c_int()
        result = self.libcuda.cuDeviceGet(byref(dev), 0)  # todo: support multi-gpu
        if result != 0:
            raise RuntimeError(f"cuDeviceGet failed: {result}")

        self.device_handle = dev.value

        ctx = c_void_p()
        result = self.libcuda.cuCtxCreate(byref(ctx), 0, self.device_handle)
        if result != 0:
            raise RuntimeError(f"cuCtxCreate failed: {result}")

        self.ctx = ctx

    def __del__(self):
        if self.ctx:
            self.libcuda.cuCtxDestroy(self.ctx)
            self.context = None


cuda = CUDA()


class LangFlavor(kernel.LangFlavor):
    @classmethod
    def indent_size(cls) -> int:
        return 4

    @classmethod
    def line_term(cls) -> str:
        return ";"

    @classmethod
    def kern_qualifier(cls) -> str:
        return 'extern "C" __global__'

    @classmethod
    def block_start(cls) -> str:
        return "{"

    @classmethod
    def block_end(cls) -> str:
        return "}"

    @classmethod
    def if_cond_start(cls) -> str:
        return "("

    @classmethod
    def if_cond_end(cls) -> str:
        return ")"

    @classmethod
    def loop_cond_start(cls) -> str:
        return "("

    @classmethod
    def loop_cond_end(cls) -> str:
        return ")"

    @classmethod
    def loop_cond(cls, start: str, stop: str, var: str) -> str:
        return f"int {var} = {start}; {var} < {stop}; {var}++"

    @classmethod
    def typegen(cls, typ: kernel.VType) -> str:
        if typ.typ == kernel.VTypeCode.I32:
            tp = "int"
        elif typ.typ == kernel.VTypeCode.I64:
            tp = "long long"
        elif typ.typ == kernel.VTypeCode.F32:
            tp = "float"
        elif typ.typ == kernel.VTypeCode.VOID:
            tp = "void"

        if typ.pointer:
            tp += "*"

        return tp

    @classmethod
    def unary_op_gen(cls, operand: str, valid_operand: str, code: kernel.OpCode) -> str:
        if code == kernel.OpCode.RECIP:
            return f"1.0f / ({valid_operand} ? {operand} : 1e-6)"
        if code == kernel.OpCode.LOG:
            return f"log({valid_operand} ? {operand} : 1e-6)"
        if code == kernel.OpCode.COPY:
            return f"{operand}"

        raise RuntimeError(f"unhandled code {code}")

    @classmethod
    def binary_op_gen(cls, code: kernel.OpCode, loperand: str, roperand: str, valid_loperand: str, valid_roperand: str) -> str:
        if code == kernel.OpCode.ADD:
            return f"{loperand} + {roperand}"
        if code == kernel.OpCode.SUM:
            return f"{loperand} + {roperand}"
        if code == kernel.OpCode.MUL:
            return f"{loperand} * {roperand}"
        if code == kernel.OpCode.POW:
            return f"powf({loperand}, {roperand})"

        raise RuntimeError(f"unhandled code {code}")

    @classmethod
    def grid_dim(self, dim: str) -> str:
        return f"gridDim.{dim}"

    @classmethod
    def block_idx(self, dim: str) -> str:
        return f"blockIdx.{dim}"

    @classmethod
    def block_dim(self, dim: str) -> str:
        return f"blockDim.{dim}"

    @classmethod
    def thread_idx(self, dim: str) -> str:
        return f"threadIdx.{dim}"


class CodeGenerator(kernel.CodeGenerator):
    def __init__(self):
        super().__init__(LangFlavor)


class Device(device.Device):
    def allocate(self, length: int):
        size = length * sizeof(c_float)  # byte size
        ptr = c_void_p()
        result = cuda.libcuda.cuMemAlloc(byref(ptr), size)
        if result != 0:
            raise RuntimeError(f"cuMemAlloc failed: {result}")

        return device.DeviceMemoryBuffer(ptr, length, size)

    def free(self, dev_buff: device.DeviceMemoryBuffer):
        cuda.libcuda.cuMemFree(dev_buff.ptr)

    def copy_to_device(self, cpu_buff: device.CPUMemoryBuffer, dev_buff: device.DeviceMemoryBuffer):
        result = cuda.libcuda.cuMemcpyHtoD(dev_buff.ptr, (c_float * dev_buff.length)(*cpu_buff.raw), dev_buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyHtoD failed: {result}")

    def copy_from_device(self, dev_buff: device.DeviceMemoryBuffer, cpu_buff: device.CPUMemoryBuffer):
        out = (c_float * dev_buff.length)()
        result = cuda.libcuda.cuMemcpyDtoH(out, dev_buff.ptr, dev_buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed: {result}")

        cpu_buff.raw = [out[i] for i in range(dev_buff.length)]


class PTXCompiler:
    def __init__(self, dir="/tmp"):
        self.dir = dir

    def compile_and_get_ptx_src(self, kern_src):
        t = int(time.time())
        kern_src_path = f"{self.dir}/kern_{t}.cu"
        with open(kern_src_path, "w") as f:
            f.write(kern_src)

        ptx_path = f"{self.dir}/kern_{t}.ptx"
        result = os.system(f"nvcc --ptx {kern_src_path} -o {ptx_path}")
        if result != 0:
            raise RuntimeError(f"ptx compilation failed: {result}")

        os.remove(kern_src_path)

        with open(ptx_path, "rb") as f:
            ptx_src = f.read()

        os.remove(ptx_path)

        return ptx_src


class KernelManager(kernel.KernelManager):
    def __init__(self, dir="/tmp"):
        self.ptx_compiler = PTXCompiler(dir)
        super().__init__()

    def load_kern_ptr(self, kerns: list[kernel.Kernel]) -> list[c_void_p]:
        whole_src = "\n\n".join([k.src for k in kerns])
        ptx_src = self.ptx_compiler.compile_and_get_ptx_src(whole_src)

        mod = c_void_p()
        result = cuda.libcuda.cuModuleLoadData(byref(mod), ptx_src)
        if result != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {result}")

        fps = []
        for k in kerns:
            fp = c_void_p()
            result = cuda.libcuda.cuModuleGetFunction(byref(fp), mod, k.name.encode("utf-8"))
            if result != 0:
                raise RuntimeError(f"cuModuleGetFunction failed: {result}")
            fps.append(fp)

        return fps

    def invoke(self, kern_name: str, grid: int | tuple[int], block: int | tuple[int], params: tuple[typing.Any]):
        def extract(p: int | tuple[int]):
            if type(p) == int:
                return (p, 1, 1)
            elif type(p) == list or type(p) == tuple:
                return (p[0] if 0 < len(p) else 1, p[1] if 1 < len(p) else 1, p[2] if 2 < len(p) else 1)

        kernel_params = (c_void_p * len(params))()
        for i, p in enumerate(params):
            if type(p) is device.DeviceMemoryBuffer:
                kernel_params[i] = cast(byref(p.ptr), c_void_p)
            elif type(p) is int:
                kernel_params[i] = cast(pointer(c_int(p)), c_void_p)
            else:
                raise TypeError(f"cannot handle {type(p)} as kernel parameter")

        kern = self.get_kern(kern_name)
        if kern is None:
            raise RuntimeError(f"no kernel found by name: {kern_name}")

        result = cuda.libcuda.cuLaunchKernel(
            kern.func_pointer,
            *extract(grid),
            *extract(block),
            0,
            None,
            kernel_params,
            None,
        )
        if result != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {result}")

        result = cuda.libcuda.cuCtxSynchronize()
        if result != 0:
            raise RuntimeError(f"cuCtxSynchronize failed: {result}")
