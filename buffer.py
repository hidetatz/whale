import sys
from dataclasses import dataclass

class CPUBuff:
    def __init__(self, val=None):
        self.val = val

    def __repr__(self):
        if not self.val: return "None"
        if len(self.val) < 4: return f"{self.val} ({len(self.val)} items)"
        return f"[{self.val[0]}, {self.val[1]}, ... {self.val[-1]}] (len={len(self.val)})"

class DevBuff:
    def __init__(self, ptr=None):
        self.ptr = ptr

    def __repr__(self): return "copied" if self.ptr else "None"

    def __del__(self):
        backend = sys.modules.get("backend")
        if backend and self.ptr: backend.free(self.ptr)

@dataclass
class Buffer:
    dtype: dtype
    length: int
    cpu: CPUBuff
    dev: DevBuff
