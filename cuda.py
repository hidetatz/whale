from ctypes import CDLL, byref, cast, c_void_p, c_int, c_uint, c_float, sizeof
from dataclasses import dataclass
import os
import time

@dataclass
class CUDAKernel:
    mod: any
    func: any

@dataclass
class CUDADeviceBuffer:
    ptr: any  # pointer on device memory
    length: int  # array length
    size: int  # byte size

class CUDA:
    def __init__(self, ptx_dir="/tmp"):
        self.ptx_dir = ptx_dir

        try:
            self.libcuda = CDLL('libcuda.so')
        except OSError as e:
            raise RuntimeError(f"libcuda.so is not found: {e}")

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

    # allocate device memory
    def memalloc_float(self, length):
        size = length * sizeof(c_float)  # byte size
        ptr = c_void_p()
        result = self.libcuda.cuMemAlloc(byref(ptr), size)
        if result != 0:
            raise RuntimeError(f"cuMemAlloc failed: {result}")

        return CUDADeviceBuffer(ptr, length, size)

    def memcpyHtoD(self, buff, input):
        result = self.libcuda.cuMemcpyHtoD(buff.ptr, (c_float * buff.length)(*input), buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyHtoD failed: {result}")

    def memcpyDtoH(self, buff):
        out = (c_float * buff.length)()
        result = self.libcuda.cuMemcpyDtoH(out, buff.ptr, buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed: {result}")

        return [out[i] for i in range(buff.length)]

    # free device memory
    def free(self, buff):
        self.libcuda.cuMemFree(buff.input)
        del buff

    def load_kernel(self, kern_src_path, kern_name):
        t = int(time.time())
        ptx_path = f"{self.ptx_dir}/kern_{t}.ptx"
        result = os.system(f"nvcc --ptx {kern_src_path} -o {ptx_path}")
        if result != 0:
            raise RuntimeError(f"ptx compilation failed: {result}")

        try:
            with open(ptx_path, "rb") as f:
                ptx = f.read()
        except IOError:
            raise RuntimeError(f"loading ptx failed: {ptx_path}")
        
        mod = c_void_p()
        result = self.libcuda.cuModuleLoadData(byref(mod), ptx)
        if result != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {result}")

        func = c_void_p()
        result = self.libcuda.cuModuleGetFunction(byref(func), mod, kern_name.encode("utf-8"))
        if result != 0:
            raise RuntimeError(f"cuModuleGetFunction failed: {result}")

        os.remove(ptx_path)

        return CUDAKernel(mod, func)

    def call_kernel(self, kern, grid, block, params):
        def extract(p):
            if type(p) == int:
                return p, 1, 1
            elif type(p) == list or type(p) == tuple:
                return p[0], p[1] if 1 < len(p) else 1, p[2] if 2 < len(p) else 1

        gridx, gridy, gridz = extract(grid)
        blockx, blocky, blockz = extract(block)
        kernel_params = (c_void_p * len(params))(*[cast(byref(p.ptr), c_void_p) for p in params])
        result = self.libcuda.cuLaunchKernel(
            kern.func,
            gridx, gridy, gridz, blockx, blocky, blockz,
            0, None, kernel_params, None
        )
        if result != 0:
            raise RuntimeError(f"cuLaunchKernel failed: {result}")
        
        result = self.libcuda.cuCtxSynchronize()
        if result != 0:
            raise RuntimeError(f"cuCtxSynchronize failed: {result}")

    def __del__(self):
        if self.ctx:
            self.libcuda.cuCtxDestroy(self.ctx)
            self.context = None

if __name__ == "__main__":
    cuda = CUDA()

    try:
        kern = cuda.load_kernel("relu.cu", "kern")

        input = cuda.memalloc_float(4)
        cuda.memcpyHtoD(input, [-1.0, 0.0, 1.0, 2.0])

        output = cuda.memalloc_float(4)

        cuda.call_kernel(kern, 1, 4, (input, output))
        result = cuda.memcpyDtoH(output)

        print(f"Output: {result}")

    except Exception as e:
        print(f"error: {e}")

    finally:
        del cuda
