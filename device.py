from dataclasses import dataclass

@dataclass
class PythonBuffer:
    value: any  # list

@dataclass
class GPUBuffer:
    ptr: any  # pointer on device memory
    length: int  # array length
    size: int  # byte size

class Renderer:
    def render_kern_add(self): raise NotImplementedError()
    def render_kern_mul(self): raise NotImplementedError()

class Allocator:
    def allocate(self, length): raise NotImplementedError()
    def free(self, gpu_buff): raise NotImplementedError()
    def copy_to_device(self, py_buff, gpu_buff): raise NotImplementedError()
    def copy_from_device(self, gpu_buff): raise NotImplementedError()

@dataclass
class Kernel:
    name: str
    func_pointer: any  # on C side

@dataclass
class KernelSrc:
    src: str
    name: str

class KernelManager:
    def __init__(self):
        self.kerns = []

    def load(self, srcs):
        fps = self.load_kern_ptr(srcs)
        self.kerns.extend([Kernel(src.name, fp) for src, fp in zip(srcs, fps)])

    def get_kern(self, name):
        for k in self.kerns:
            if k.name == name:
                return k

    def load_kern_ptr(self, *params): raise NotImplementedError()
    def invoke(self, kern_name, grid, block, params): raise NotImplementedError()
