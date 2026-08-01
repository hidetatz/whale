import os
from abc import ABC, abstractmethod

DEBUG = os.environ.get("WHALE_DEBUG", "0") == "1"

def dbg_raw(msg: str):
    print(f"{msg}")

def dbg(msg: str):
    dbg_raw(f"[debug] {msg}")

def dbg(msg: str):
    dbg_raw(f"[debug] {msg}")

class DebuggableTree(ABC):
    @abstractmethod
    def debug_str(self): ...
    @abstractmethod
    def debug_children(self): ...

def dbg_tree(t: DebuggableTree):
    def f(_t, depth=0, prefix="        ", last=True):
        connector = "└─ " if last else "├─ "
        line = prefix + (connector if depth > 0 else "") + _t.debug_str()
        children = _t.debug_children()
        child_prefix = prefix + ("   " if last else "│  ")
        child_lines = [f(c, depth+1, child_prefix, i==len(children)-1) for i, c in enumerate(children)]
        return "\n".join([line] + child_lines)

    dbg_raw(f(t))

def dbg_kernel(kern: str):
    for l in kern.splitlines():
        dbg_raw("        " + l)
