import os
import time
from ctypes import CDLL, byref, c_float, c_int, c_uint, c_void_p, cast, pointer, sizeof

import device
import kernel


class CUDA:
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")

        if self.libcuda.cuInit(0) != 0:
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

    def header(self, code: kernel.OpCode) -> str:
        return "#include <cmath>" if code == kernel.OpCode.POW else ""

    def kern_qualifier(self, _) -> str:
        return 'extern "C" __global__ void'

    def kern_param_ident(self, pname: str, typ=int | float, pointer=False, const=False, memory: str = "host") -> str:
        tp = "int" if typ is int else "float"
        if pointer:
            tp += "*"
        return f"{tp} {pname}"

    def thread_idx_expr(self, ndim: int, params: int) -> list[str]:
        xyz = [
            "int x = blockIdx.x * blockDim.x + threadIdx.x;",
            "int y = blockIdx.y * blockDim.y + threadIdx.y;",
            "int z = blockIdx.z * blockDim.z + threadIdx.z;",
        ][: ndim if ndim else 1]

        def toidx(ndim: int, pref: str):
            if ndim == 0 or ndim == 1:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0;"
            elif ndim == 2:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0 + y * {pref}_stride1;"
            elif ndim == 3:
                return f"int {pref}_idx = {pref}_offset + x * {pref}_stride0 + y * {pref}_stride1 + z * {pref}_stride2;"

        idxs = [toidx(ndim, "dst")]
        for i in range(params):
            idxs.append(toidx(ndim, f"src_{i}"))

        return xyz + idxs

    def kern_body(self, code: kernel.OpCode, ndim: int) -> list[str]:
        if code == kernel.OpCode.RECIP:
            return ["dst[dst_idx] = 1.0f / src_0[src_0_idx];"]
        if code == kernel.OpCode.ADD:
            return ["dst[dst_idx] = src_0[src_0_idx] + src_1[src_1_idx];"]
        if code == kernel.OpCode.MUL:
            return ["dst[dst_idx] = src_0[src_0_idx] * src_1[src_1_idx];"]
        if code == kernel.OpCode.POW:
            return ["dst[dst_idx] = powf(src_0[src_0_idx], src_1[src_1_idx]);"]

        raise RuntimeError(f"kern body is not defined on op {code}")


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
        del gpu_buff

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

    def load_kern_ptr(self, kerns):
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

    def invoke(self, kern_name: str, grid: int | tuple[int], block: int | tuple[int], params: tuple[any]):
        def extract(p: int | tuple[int]):
            if type(p) == int:
                return (p, 1, 1)
            elif type(p) == list or type(p) == tuple:
                return (p[0], p[1] if 1 < len(p) else 1, p[2] if 2 < len(p) else 1)

        kernel_params = (c_void_p * len(params))()
        for i, p in enumerate(params):
            if type(p) is device.DeviceMemoryBuffer:
                kernel_params[i] = cast(byref(p.ptr), c_void_p)
            elif type(p) is int:
                kernel_params[i] = cast(pointer(c_int(p)), c_void_p)
            else:
                raise TypeError(f"cannot handle {type(p)} as kernel parameter")

        result = cuda.libcuda.cuLaunchKernel(
            self.get_kern(kern_name).func_pointer,
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
