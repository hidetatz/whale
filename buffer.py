from dataclasses import dataclass

class CPUBuff:
    def __init__(self, val=None, dtype=None):
        self.val = val
        self.dtype = dtype
        if val and dtype is None:
            assert type(val[0]) is int or type(val[0]) is float
            self.dtype = int64 if type(val[0]) is int else float64

    def __repr__(self):
        if not self.val: return "None"
        if len(self.val) < 4: return f"{self.val} ({len(self.val)} items)"
        return f"[{self.val[0]}, {self.val[1]}, ... {self.val[-1]}] ({len(self.val)} items)"

class DevBuff:
    def __init__(self, ptr=None, dtype=None):
        self.ptr = ptr
        self.dtype = dtype

    def __repr__(self): return "copied" if self.ptr else "None"

@dataclass
class Buffer:
    cpu: CPUBuff
    dev: DevBuff
