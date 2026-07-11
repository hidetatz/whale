from dataclasses import dataclass
from dtype import int64, float64

class CPUBuff:
    def __init__(self, val=None):
        self.val = val

    def __repr__(self):
        if not self.val: return "None"
        if len(self.val) < 4: return f"{self.val} ({len(self.val)} items)"
        return f"[{self.val[0]}, {self.val[1]}, ... {self.val[-1]}] ({len(self.val)} items)"

class DevBuff:
    def __init__(self, ptr=None):
        self.ptr = ptr

    def __repr__(self): return "copied" if self.ptr else "None"

@dataclass
class Buffer:
    dtype: dtype
    length: int
    cpu: CPUBuff
    dev: DevBuff
