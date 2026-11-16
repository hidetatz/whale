import os
import subprocess
import tempfile
from ctypes import byref, cast, sizeof, c_void_p, c_int, c_double, c_float
from ctypes import CDLL

import compiler
import executor
import langspec
from datatype import int64
from kernel import Kernel

class CUDALangSpec(langspec.CCompatibleLangSpec):
    def kern_start(self, name, arg_names, arg_types): return f"extern \"C\" __global__ void {name}({", ".join([f'{self.typename(tp)}* {nm}' for nm, tp in zip(arg_names, arg_types)])}) {{"
    def scatter_add(self, buf, addr, val): return f"atomicAdd(&{buf}[{addr}], {val});"
    def parallel_loop_start(self, index, start, end, step): raise RuntimeError("cuda backend cannot handle parallel loop: use gpu_blocks/threads instead!")
    def vectorized_loop_start(self, index, start, end, step): raise RuntimeError("cuda backend cannot handle vectorized loop: use gpu_blocks/threads instead!")
    def unrolled_loop_start(self, index, start, end, step, factor):
        return [
            f"#pragma unroll {factor}",
            self.sequential_loop_start(index, start, end, step),
        ]
    def gpu_block_index(self, index, start, end, step, idx): return self.init(int64, index, f"blockIdx.{idx}")
    def gpu_thread_index(self, index, start, end, step, idx): return self.init(int64, index, f"threadIdx.{idx}")

CUBLAS_OP_N = 0

class CUBlas:
    def __init__(self):
        import glob
        candidates = ["libcublas.so"] + glob.glob("/usr/local/cuda*/targets/*/lib/libcublas.so")
        for name in candidates:
            try:
                self.lib = CDLL(name); break
            except OSError: pass
        else:
            raise RuntimeError("libcublas.so not found")
        self.handle = c_void_p()
        result = self.lib.cublasCreate_v2(byref(self.handle))
        if result != 0: raise RuntimeError(f"cublasCreate_v2: {result}")

    def dgemm(self, m, n, k, a_ptr, b_ptr, c_ptr):
        # row-major C(M×N) = A(M×K) @ B(K×N)
        # → cublas col-major: C^T(N×M) = B^T(N×K) @ A^T(K×M)
        # → cublasDgemm(N, N, N, M, K, B_ptr, N, A_ptr, K, C_ptr, N)
        alpha, beta = c_double(1.0), c_double(0.0)
        r = self.lib.cublasDgemm_v2(self.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, byref(alpha), b_ptr, n, a_ptr, k, byref(beta), c_ptr, n)
        if r != 0: raise RuntimeError(f"cublasDgemm_v2: {r}")

    def sgemm(self, m, n, k, a_ptr, b_ptr, c_ptr):
        alpha, beta = c_float(1.0), c_float(0.0)
        r = self.lib.cublasSgemm_v2(self.handle, CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k, byref(alpha), b_ptr, n, a_ptr, k, byref(beta), c_ptr, n)
        if r != 0: raise RuntimeError(f"cublasSgemm_v2: {r}")

    def __del__(self):
        if self.handle:
            self.lib.cublasDestroy_v2(self.handle)
            self.handle = None

class CUDA:
    def __init__(self):
        self.libcuda = CDLL("libcuda.so")
        self.libcudart = CDLL("libcudart.so")
        self.exec("cuInit", 0)

        dev = c_int()
        self.exec("cuDeviceGet", byref(dev), 0)
        self.device_handle = dev.value

        # initialize the CUDA runtime primary context via cudaFree(NULL);
        # this makes the same context usable by both driver API (cuModuleLoadData,
        # cuLaunchKernel) and runtime API (cudaMalloc, cuBLAS)
        self.exec_rt("cudaFree", None)
        ctx = c_void_p()
        self.exec("cuCtxGetCurrent", byref(ctx))
        self.ctx = ctx

    def exec(self, f, *args):
        fn = getattr(self.libcuda, f)
        result = fn(*args)
        if result != 0: raise RuntimeError(f"{f}: {result}")

    def exec_rt(self, f, *args):
        fn = getattr(self.libcudart, f)
        result = fn(*args)
        if result != 0: raise RuntimeError(f"{f}: {result}")

class CUDACompiler(compiler.Compiler):
    def __init__(self, cuda):
        self.cuda = cuda

    # exec settings
    def compile(self, name, code):
        with tempfile.NamedTemporaryFile(suffix=".ptx", delete=False) as f: ptx = f.name
        subprocess.run(["nvcc", "-ptx", "-x", "cu", "-", "-o", ptx], input=code, check=True, text=True)
        with open(f"{ptx}", "rb") as f: ptx_src = f.read()
        os.remove(ptx)
        # load ptx as module
        mod = c_void_p()
        self.cuda.exec("cuModuleLoadData", byref(mod), ptx_src)
        ptr = c_void_p()
        self.cuda.exec("cuModuleGetFunction", byref(ptr), mod, name.encode("utf-8"))
        return Kernel(ptr)

class CUDAExecutor(executor.GPUExecutor):
    def __init__(self, cuda, cublas=None):
        self.cuda = cuda
        self.cublas = cublas

    def cublas_matmul(self, m, n, k, a_buf, b_buf, c_buf, dtype):
        cb = self.cublas
        a_ptr, b_ptr, c_ptr = a_buf.dev.ptr, b_buf.dev.ptr, c_buf.dev.ptr
        from datatype import Float32
        if isinstance(dtype, Float32):
            cb.sgemm(m, n, k, a_ptr, b_ptr, c_ptr)
        else:
            cb.dgemm(m, n, k, a_ptr, b_ptr, c_ptr)

    def execute(self, kern: Kernel, params: list[buffer.Buffer], grid: tuple[int, int, int], block: tuple[int, int, int]):
        kern_ptr = kern.bin
        c_params = (c_void_p * len(params))()
        for i, p in enumerate(params):
            c_params[i] = cast(byref(p.dev.ptr), c_void_p)

        self.cuda.exec("cuLaunchKernel", kern_ptr, *grid, *block,
            0, # sharedMemBytes
            None, # hStream
            c_params,
            None, # extra
        )

    _CUDA_MEMCPY_H2D = 1
    _CUDA_MEMCPY_D2H = 2

    def memalloc(self, length, ctype):
        ptr = c_void_p()
        self.cuda.exec_rt("cudaMalloc", byref(ptr), sizeof(ctype) * length)
        return ptr

    def free(self, ptr):
        self.cuda.exec_rt("cudaFree", ptr)

    def memcpy_htod(self, dst, src, length, ctype):
        self.cuda.exec_rt("cudaMemcpy", dst, (ctype * length)(*src), sizeof(ctype) * length, self._CUDA_MEMCPY_H2D)

    def memzero(self, ptr, length, ctype):
        self.cuda.exec_rt("cudaMemset", ptr, 0, sizeof(ctype) * length)

    def memcpy_dtoh(self, src, length, ctype):
        self.cuda.exec_rt("cudaDeviceSynchronize")
        out = (ctype * length)()
        self.cuda.exec_rt("cudaMemcpy", out, src, sizeof(ctype) * length, self._CUDA_MEMCPY_D2H)
        return [out[i] for i in range(length)]
