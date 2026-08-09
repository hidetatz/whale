import math
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Literal

class LoopKind(IntEnum):
    Spatial = auto()
    Reduce = auto()

    def __str__(self): return self.name

class LoopVar:
    def __init__(self, name: str, extent: int):
        self.name = name
        self.extent = extent

    def __str__(self): return f"{self.name}:{self.extent}"
    def __repr__(self): return f"LoopVar(name={self.name}, extent={self.extent})"

@dataclass
class SplitLoop:
    orig: LoopVar
    o: LoopVar
    i: LoopVar
    # split factor: inner loop runs from zero to the factor.
    factor: int
    tail_guard_required: bool

    def __str__(self): return f"SplitLoop: {self.orig} -> {self.o}, {self.i}"

class LoopExec:
    def __str__(self): return self.__class__.__name__

class Sequential(LoopExec): pass
class Parallel(LoopExec): pass
class Vectorize(LoopExec): pass

class Unroll(LoopExec):
    def __init__(self, factor: int):
        self.factor = factor
    def __str__(self): return super().__str__() + f"({self.factor})"

class GPUBlock(LoopExec):
    def __init__(self, index: Literal["x", "y", "z"]):
        self.index = index
    def __str__(self): return super().__str__() + f"({self.index})"

class GPUThread(LoopExec):
    def __init__(self, index: Literal["x", "y", "z"]):
        self.index = index
    def __str__(self): return super().__str__() + f"({self.index})"

class LoopSched:
    def __init__(self, lv: LoopVar, kind: LoopKind, exec: LoopExec):
        self.lv = lv
        self.kind = kind
        self.exec = exec

    def __str__(self): return f"{self.lv} ({self.kind}:{self.exec})"

class Schedule:
    def __init__(self, loops: list[LoopSched], splits: list[SplitLoop]):
        self.loops = loops
        self.splits = splits

class SchedulerFunc:
    def __init__(self, efunc):
        spatials = [LoopSched(LoopVar(lv.name, lv.extent), LoopKind.Spatial, Sequential()) for lv in efunc.out_loops]
        reduces = [LoopSched(LoopVar(lv.name, lv.extent), LoopKind.Reduce, Sequential()) for lv in efunc.reduced_vars()]
        self.loops = spatials + reduces
        self.splits = []

    def split(self, lv, factor):
        outer = LoopVar(f"{lv.name}__o", math.ceil(lv.extent / factor))
        inner = LoopVar(f"{lv.name}__i", factor)

        tail_guard_required = lv.extent % factor != 0
        self.splits.append(SplitLoop(lv, outer, inner, factor, tail_guard_required))
        idx = self._index(lv)
        kind = self.loops[idx].kind
        self.loops[idx:idx+1] = [LoopSched(outer, kind, Sequential()), LoopSched(inner, kind, Sequential())]
        return outer, inner

    def reorder(self, *lvs):
        indices = sorted([self._index(lv) for lv in lvs], reverse=True)
        for i, lv in zip(indices, lvs):
            self.loops[i].lv = lv
        return self

    def parallel(self, lv):
        self._find(lv).exec = Parallel()
        return self

    def vectorize(self, lv):
        self._find(lv).exec = Vectorize()
        return self

    def unroll(self, lv, factor):
        self._find(lv).exec = Unroll(factor)
        return self

    def tile(self, x, y, xo, yo, xi, yi, xfactor, yfactor):
        self.split(x, xo, xi, xfactor)
        self.split(y, yo, yi, yfactor)
        self.reorder(xi, yi, xo, yo)
        return self

    def gpu_blocks(self, lv, index):
        self._find(lv).exec = GPUBlock(index)
        return self

    def gpu_blocks_x(self, lv): return self.gpu_blocks(lv, "x")
    def gpu_blocks_y(self, lv): return self.gpu_blocks(lv, "y")
    def gpu_blocks_z(self, lv): return self.gpu_blocks(lv, "z")

    def gpu_threads(self, lv, index):
        self._find(lv).exec = GPUThread(index)
        return self

    def gpu_threads_x(self, lv): return self.gpu_threads(lv, "x")
    def gpu_threads_y(self, lv): return self.gpu_threads(lv, "y")
    def gpu_threads_z(self, lv): return self.gpu_threads(lv, "z")

    def schedule(self): return Schedule(self.loops, self.splits)

    def spatial_loops(self):
        return [l.lv for l in self.loops if l.kind == LoopKind.Spatial]

    def _index(self, lv):
        for i, l in enumerate(self.loops):
            if l.lv == lv: return i
        raise RuntimeError("loopvar not found in loops")

    def _find(self, lv):
        return self.loops[self._index(lv)]

class AutoScheduler:
    @staticmethod
    def schedule_cpu(sf):
        spatials = sf.spatial_loops()

        # The outermost spatial loop: parallelize
        if spatials: sf.parallel(spatials[0])

        # The innermost spatial loop: vectorize
        if 1 < len(spatials): sf.vectorize(spatials[-1])

    @staticmethod
    def schedule_gpu(sf):
        spatials = sf.spatial_loops()
        if not spatials: return

        block_size = 256

        n = len(spatials)

        if n == 1:
            io, ii = sf.split(spatials[0], block_size)
            sf.gpu_blocks_x(io).gpu_threads_x(ii)

        elif n == 2:
            io, ii = sf.split(spatials[1], block_size)
            sf.gpu_blocks_x(spatials[0]).gpu_blocks_y(io).gpu_threads_x(ii)

        elif n == 3:
            io, ii = sf.split(spatials[2], block_size)
            sf.gpu_blocks_x(spatials[0]).gpu_blocks_y(spatials[1]).gpu_blocks_z(io).gpu_threads_x(ii)

        else:
            io, ii = sf.split(spatials[-1], block_size)
            sf.gpu_blocks_x(io).gpu_threads_x(ii)

def schedule(funcs, gpu, scheduler=AutoScheduler):
    scheds = []
    for f in funcs:
        sf = SchedulerFunc(f)
        if gpu:
            scheduler.schedule_gpu(sf)
        else:
            scheduler.schedule_cpu(sf)

        scheds.append(sf.schedule())

    return scheds
