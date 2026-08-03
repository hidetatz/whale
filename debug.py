import os
import typing

DEBUG = os.environ.get("WHALE_DEBUG", "0") == "1"

def dbg_raw(msg: str):
    print(f"{msg}")

def dbg(msg: str):
    dbg_raw(f"[debug] {msg}")

class TreeNode(typing.Protocol):
    def inputs(self): ...

def dbg_tree(n: TreeNode):
    def f(_n: tree.Node, depth=0, prefix="        ", last=True):
        connector = "└─ " if last else "├─ "
        line = prefix + (connector if depth > 0 else "") + _n.__str__()
        inputs = _n.inputs()
        inp_prefix = prefix + ("   " if last else "│  ")
        inp_lines = [f(c, depth+1, inp_prefix, i==len(inputs)-1) for i, c in enumerate(inputs)]
        return "\n".join([line] + inp_lines)

    dbg_raw(f(n))

def dbg_schedule(s: 'sched.Schedule'):
    dbg_raw(f"        Schedule:")
    for l in s.loops:
        dbg_raw(f"          {l}")
    for ss in s.splits:
        dbg_raw(f"          {ss}")

def dbg_kernel(kern: str):
    for l in kern.splitlines():
        dbg_raw("        " + l)
