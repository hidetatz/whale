from ctypes import CDLL, byref, cast, c_void_p, c_int, c_uint, c_float, sizeof, pointer
import os
import time

import device


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


class Renderer(device.Renderer):
    def render_kernel(self, name, params, body):
        return f'extern "C" __global__ void {name}({", ".join([f"float* {param}" for param in params])}) {{\n    {"\n    ".join(body)}\n}}'

    def render_kernel2(self, name, params, body):
        return f'extern "C" __global__ void {name}({", ".join(params)}) {{\n    {"\n    ".join(body)}\n}}'

    def render_linearized_idx_expr(self, shape):
        match len(shape):
            case 0 | 1:
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "int idx = offset + x * stride0;",
                ]
            case 2:
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "int y = blockIdx.y * blockDim.y + threadIdx.y;",
                    "int idx = offset + x * stride0 + y * stride1;",
                ]
            case 3:
                return [
                    "int x = blockIdx.x * blockDim.x + threadIdx.x;",
                    "int y = blockIdx.y * blockDim.y + threadIdx.y;",
                    "int z = blockIdx.z * blockDim.z + threadIdx.z;",
                    "int idx = offset + x * stride0 + y * stride1 + z * stride2;",
                ]

        # todo: support
        raise RuntimeError("4 or more dimension tensor is not supported yet")

    def render_kern_add(self, shape, strides, offset):
        name = f"add_{'_'.join(map(str, shape))}"
        match len(shape):
            case 0 | 1:
                params = ("int offset", "int stride0", "float* l", "float* r", "float* result")
            case 2:
                params = ("int offset", "int stride0", "int stride1", "float* l", "float* r", "float* result")
            case 3:
                params = ("int offset", "int stride0", "int stride1", "int stride2", "float* l", "float* r", "float* result")
        body = self.render_linearized_idx_expr(shape) + ["result[idx] = l[idx] + r[idx];"]
        kern = self.render_kernel2(name, params, body)
        return kern, name

    def render_kern_mul(self, shape, strides, offset):
        name = f"mul_{'_'.join(map(str, shape))}"
        match len(shape):
            case 0 | 1:
                params = ("int offset", "int stride0", "float* l", "float* r", "float* result")
            case 2:
                params = ("int offset", "int stride0", "int stride1", "float* l", "float* r", "float* result")
            case 3:
                params = ("int offset", "int stride0", "int stride1", "int stride2", "float* l", "float* r", "float* result")
        body = self.render_linearized_idx_expr(shape) + ["result[idx] = l[idx] * r[idx];"]
        return self.render_kernel2(name, params, body), name

    def render_kern_recip(self, shape, strides, offset):
        name = f"recip_{'_'.join(map(str, shape))}"
        match len(shape):
            case 0 | 1:
                params = ("int offset", "int stride0", "float* r", "float* result")
            case 2:
                params = ("int offset", "int stride0", "int stride1", "float* r", "float* result")
            case 3:
                params = ("int offset", "int stride0", "int stride1", "int stride2", "float* r", "float* result")
        body = self.render_linearized_idx_expr(shape) + ["result[idx] = 1.0f / r[idx];"]
        return self.render_kernel2(name, params, body), name
        # return self.render_kernel(
        #     "recip",
        #     ("x", "result"),
        #     [
        #         "int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        #         "result[idx] = 1.0f / x[idx];",
        #     ],
        # )

    def render_kern_pow(self, shape, strides, offset):
        name = f"pow_{'_'.join(map(str, shape))}"
        match len(shape):
            case 0 | 1:
                params = ("int offset", "int stride0", "float* l", "float* r", "float* result")
            case 2:
                params = ("int offset", "int stride0", "int stride1", "float* l", "float* r", "float* result")
            case 3:
                params = ("int offset", "int stride0", "int stride1", "int stride2", "float* l", "float* r", "float* result")
        body = self.render_linearized_idx_expr(shape) + ["result[idx] = powf(l[idx], r[idx]);"]
        return "#include <cmath>\n\n" + self.render_kernel2(name, params, body), name
        # inc = "#include <cmath>"
        # kern = self.render_kernel(
        #     "power",
        #     ("l", "r", "result"),
        #     [
        #         "int idx = blockIdx.x * blockDim.x + threadIdx.x;",
        #         "result[idx] = powf(l[idx], r[idx]);",
        #     ],
        # )
        # return inc + "\n\n" + kern


class Allocator(device.Allocator):
    def allocate(self, length):
        size = length * sizeof(c_float)  # byte size
        ptr = c_void_p()
        result = cuda.libcuda.cuMemAlloc(byref(ptr), size)
        if result != 0:
            raise RuntimeError(f"cuMemAlloc failed: {result}")

        return device.GPUBuffer(ptr, length, size)

    def free(self, gpu_buff):
        cuda.libcuda.cuMemFree(gpu_buff.input)
        del gpu_buff

    def copy_to_device(self, py_buff, gpu_buff):
        result = cuda.libcuda.cuMemcpyHtoD(gpu_buff.ptr, (c_float * gpu_buff.length)(*py_buff.value), gpu_buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyHtoD failed: {result}")

    def copy_from_device(self, gpu_buff, py_buff):
        out = (c_float * gpu_buff.length)()
        result = cuda.libcuda.cuMemcpyDtoH(out, gpu_buff.ptr, gpu_buff.size)
        if result != 0:
            raise RuntimeError(f"cuMemcpyDtoH failed: {result}")

        py_buff.value = [out[i] for i in range(gpu_buff.length)]


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


class KernelManager(device.KernelManager):
    def __init__(self, dir="/tmp"):
        self.ptx_compiler = PTXCompiler(dir)
        super().__init__()

    def load_kern_ptr(self, srcs):
        whole_src = "\n\n".join([s.src for s in srcs])
        ptx_src = self.ptx_compiler.compile_and_get_ptx_src(whole_src)

        mod = c_void_p()
        result = cuda.libcuda.cuModuleLoadData(byref(mod), ptx_src)
        if result != 0:
            raise RuntimeError(f"cuModuleLoadData failed: {result}")

        fps = []
        for src in srcs:
            fp = c_void_p()
            result = cuda.libcuda.cuModuleGetFunction(byref(fp), mod, src.name.encode("utf-8"))
            if result != 0:
                raise RuntimeError(f"cuModuleGetFunction failed: {result}")
            fps.append(fp)

        return fps

    def invoke(self, kern_name, grid, block, params):
        def extract(p):
            if type(p) == int:
                return p, 1, 1
            elif type(p) == list or type(p) == tuple:
                return p[0], p[1] if 1 < len(p) else 1, p[2] if 2 < len(p) else 1

        gridx, gridy, gridz = extract(grid)
        blockx, blocky, blockz = extract(block)
        kernel_params = (c_void_p * len(params))()
        for i, p in enumerate(params):
            if type(p) is device.GPUBuffer:
                kernel_params[i] = cast(byref(p.ptr), c_void_p)
            if type(p) is int:
                kernel_params[i] = cast(pointer(c_int(p)), c_void_p)

        # kernel_params = (c_void_p * len(params))(*[cast(byref(p.ptr), c_void_p) for p in params])
        result = cuda.libcuda.cuLaunchKernel(
            self.get_kern(kern_name).func_pointer,
            gridx,
            gridy,
            gridz,
            blockx,
            blocky,
            blockz,
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
