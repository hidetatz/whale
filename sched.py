import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Literal, override

from debug import DebuggableTree

class LoopKind(IntEnum):
    Spatial = auto()
    Reduce = auto()

    def __repr__(self): return self.name
    def __str__(self): return self.name

@dataclass
class LoopVar:
    extent: int = 0
    name: str = ""

    def __repr__(self):
        return f"{self.name}:{self.extent}"

@dataclass
class SplitLoop(DebuggableTree):
    orig: LoopVar
    o: LoopVar
    i: LoopVar
    # split factor: inner loop runs from zero to the factor.
    factor: int
    tail_guard_required: bool

    @override
    def debug_str(self): return f"SplitLoop: {self.orig} -> {self.o}, {self.i}"
    @override
    def debug_children(self): return []

@dataclass
class Sequential: pass

@dataclass
class Parallel: pass

@dataclass
class Vectorize: pass

@dataclass
class Unroll:
    factor: int

@dataclass
class GPUBlock:
    index: Literal["x", "y", "z"]

@dataclass
class GPUThread:
    index: Literal["x", "y", "z"]

type LoopExec = Sequential | Parallel | Vectorize | Unroll | GPUBlock | GPUThread

@dataclass
class LoopSched(DebuggableTree):
    lv: LoopVar
    kind: LoopKind
    exec: LoopExec

    @override
    def debug_str(self): return self.__repr__()
    @override
    def debug_children(self): return []

    def __repr__(self):
        return f"{self.lv} ({self.kind})"

@dataclass
class Schedule(DebuggableTree):
    loops: list[LoopSched]
    splits: list[SplitLoop]

    @override
    def debug_str(self): return "Schedule"
    @override
    def debug_children(self): return self.loops + self.splits

class SchedulerFunc:
    def __init__(self, efunc):
        spatials = [LoopSched(LoopVar(lv.extent, lv.name), LoopKind.Spatial, Sequential()) for lv in efunc.out_loops]
        reduces = [LoopSched(LoopVar(lv.extent, lv.name), LoopKind.Reduce, Sequential()) for lv in efunc.reduced_vars()]
        self.loops = spatials + reduces
        self.splits = []

    def split(self, lv, outer, inner, factor):
        outer.extent = math.ceil(lv.extent / factor)
        outer.name = f"{lv.name}__o"

        inner.extent = factor
        inner.name = f"{lv.name}__i"

        tail_guard_required = lv.extent % factor != 0
        self.splits.append(SplitLoop(lv, outer, inner, factor, tail_guard_required))
        idx = self._index(lv)
        kind = self.loops[idx].kind
        self.loops[idx:idx+1] = [LoopSched(outer, kind, Sequential()), LoopSched(inner, kind, Sequential())]
        return self

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

        io, ii = LoopVar(), LoopVar()

        if n == 1:
            sf.split(spatials[0], io, ii, block_size)
            sf.gpu_blocks_x(io).gpu_threads_x(ii)

        elif n == 2:
            sf.split(spatials[1], io, ii, block_size)
            sf.gpu_blocks_x(spatials[0]).gpu_blocks_y(io).gpu_threads_x(ii)

        elif n == 3:
            sf.split(spatials[2], io, ii, block_size)
            sf.gpu_blocks_x(spatials[0]).gpu_blocks_y(spatials[1]).gpu_blocks_z(io).gpu_threads_x(ii)

        else:
            sf.split(spatials[-1], io, ii, block_size)
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
