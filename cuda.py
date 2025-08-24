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


class CodeGenerator(kernel.CodeGenerator):
    def indent(self) -> str:
        return "    "

    def header(self, code: kernel.OpCode) -> list[str]:
        return ["#include <cmath>"] if code == kernel.OpCode.POW else []

    def kern_qualifier(self, _) -> str:
        return 'extern "C" __global__ void'

    def kern_param_ident(
        self, ident_name: str, typ: type[int | float] = int, const: bool = False, pointer: bool = False, memory: str = "host"
    ) -> str:
        tp = "int" if typ is int else "float"
        if pointer:
            tp += "*"
        return f"{tp} {ident_name}"

    def thread_idx_expr(self, ndim: int, params: int) -> list[str]:
        xyz = [
            "int x = blockIdx.x * blockDim.x + threadIdx.x;",
            "int y = blockIdx.y * blockDim.y + threadIdx.y;",
            "int z = blockIdx.z * blockDim.z + threadIdx.z;",
        ][: ndim if ndim else 1]

        def toidx(ndim: int, pref: str):
            if ndim == 0:
                return f"int {pref}_idx = {pref}_offset + x * 1;"
            elif ndim == 1:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0;"
            elif ndim == 2:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0 + y * {pref}_stride1;"
            elif ndim == 3:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0 + y * {pref}_stride1 + z * {pref}_stride2;"

        def idx_valid(ndim: int, pref: str):
            if ndim == 0:
                return f"int {pref}_idx_valid = 1;"
            elif ndim == 1:
                return f"int {pref}_idx_valid = {pref}_valid_area_0 <= x && x < {pref}_valid_area_1;"
            elif ndim == 2:
                return f"int {pref}_idx_valid = {pref}_valid_area_0 <= x && x < {pref}_valid_area_1 && {pref}_valid_area_2 <= y && y < {pref}_valid_area_3;"
            elif ndim == 3:
                return f"int {pref}_idx_valid = {pref}_valid_area_0 <= x && x < {pref}_valid_area_1 && {pref}_valid_area_2 <= y && y < {pref}_valid_area_3 && {pref}_valid_area_4 <= z && z < {pref}_valid_area_5;"

        def idx_val(pref: str):
            return f"float {pref}_val = {pref}_idx_valid ? {pref}[{pref}_idx] : 0.0f;"

        idxs = [toidx(ndim, "dst")]
        for i in range(params):
            idxs.append(toidx(ndim, f"src_{i}"))
        for i in range(params):
            idxs.append(idx_valid(ndim, f"src_{i}"))
        for i in range(params):
            idxs.append(idx_val(f"src_{i}"))

        return xyz + idxs

    def kern_body(self, code: kernel.OpCode, ndim: int) -> list[str]:
        if code == kernel.OpCode.RECIP:
            return ["dst[dst_idx] = 1.0f / (src_0_idx_valid ? src_0_val : 1e-6);"]
        if code == kernel.OpCode.LOG:
            return ["dst[dst_idx] = log(src_0_idx_valid ? src_0_val : 1e-6);"]
        if code == kernel.OpCode.COPY:
            return ["dst[dst_idx] = src_0_val;"]
        if code == kernel.OpCode.ADD:
            return ["dst[dst_idx] = src_0_val + src_1_val;"]
        if code == kernel.OpCode.MUL:
            return ["dst[dst_idx] = src_0_val * src_1_val;"]
        if code == kernel.OpCode.POW:
            return ["dst[dst_idx] = powf(src_0_val, src_1_val);"]

        raise RuntimeError(f"kern body is not defined on op {code}")

    # todo: this must be abstracted
    def reduce_kern_body(self, code: kernel.OpCode, ndim: int, axis: int) -> list[str]:
        if code == kernel.OpCode.SUM:
            if ndim == 0:
                raise RuntimeError("scalar cannot be reduced")

            if ndim == 1:
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "float sum = 0.0f;",
                    f"for (int x = 0; x < dim{axis}; x++) {{",
                    "    int src_0_idx = src_0_offset + x * src_0_stride0;",
                    "    int src_0_idx_valid = src_0_valid_area_0 <= x && x < src_0_valid_area_1;",
                    "    float src_0_val = src_0_idx_valid ? src_0[src_0_idx] : 0.0f;",
                    "    sum += src_0_val;",
                    "}",
                    "dst[dst_offset + x * dst_stride0] = sum;",
                ]

            if ndim == 2:
                v = "x" if axis == 0 else "y"
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "int y = blockIdx.y * blockDim.y + threadIdx.y;",
                    "float sum = 0.0f;",
                    f"for (int {v} = 0; {v} < dim{axis}; {v}++) {{",
                    "    int src_0_idx = src_0_offset + x * src_0_stride0 + y * src_0_stride1;",
                    "    int src_0_idx_valid = src_0_valid_area_0 <= x && x < src_0_valid_area_1 && src_0_valid_area_2 <= y && y < src_0_valid_area_3;",
                    "    float src_0_val = src_0_idx_valid ? src_0[src_0_idx] : 0.0f;",
                    "    sum += src_0_val;",
                    "}",
                    f"dst[dst_offset + x * dst_stride0 + y * dst_stride1] = sum;",
                ]

            if ndim == 3:
                v = "x" if axis == 0 else "y" if axis == 1 else "z"
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "int y = blockIdx.y * blockDim.y + threadIdx.y;",
                    "int z = blockIdx.z * blockDim.z + threadIdx.z;",
                    "float sum = 0.0f;",
                    f"for (int {v} = 0; {v} < dim{axis}; {v}++) {{",
                    "    int src_0_idx = src_0_offset + x * src_0_stride0 + y * src_0_stride1 + z * src_0_stride2;",
                    "    int src_0_idx_valid = src_0_valid_area_0 <= x && x < src_0_valid_area_1 && src_0_valid_area_2 <= y && y < src_0_valid_area_3 && src_0_valid_area_4 <= z && z < src_0_valid_area_5;",
                    "    float src_0_val = src_0_idx_valid ? src_0[src_0_idx] : 0.0f;",
                    "    sum += src_0_val;",
                    "}",
                    "dst[dst_offset + x * dst_stride0 + y * dst_stride1 + z * dst_stride2] = sum;",
                ]

            raise RuntimeError(f"reduce kern body is not degined on dim {ndim}")

        raise RuntimeError(f"reduce kern body is not defined on op {code}")


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
