from abc import ABC, abstractmethod

import buffer

class Executor(ABC): pass

class CPUExecutor(Executor):
    @abstractmethod
    def execute(self, kern: Kernel, params: list[buffer.Buffer]): ...

class GPUExecutor(Executor):
    @abstractmethod
    def execute(self, kern: Kernel, params: list[buffer.Buffer], grid: tuple[int, int, int], block: tuple[int, int, int]): ...
    @abstractmethod
    def memalloc(self, length, ctype): ...
    @abstractmethod
    def free(self, ptr): ...
    @abstractmethod
    def memcpy_htod(self, dst, src, length, ctype): ...
    @abstractmethod
    def memcpy_dtoh(self, src, length, ctype): ...
