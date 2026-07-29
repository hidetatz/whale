import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Literal

class LoopKind(IntEnum):
    Spatial = auto()
    Reduce = auto()

    def __repr__(self): return self.name
    def __str__(self): return self.name

@dataclass
class LoopVar:
    extent: int = 0
    name: str = ""

@dataclass
class SplitLoop:
    orig: LoopVar
    o: LoopVar
    i: LoopVar
    # split factor: inner loop runs from zero to the factor.
    factor: int

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
class LoopSched:
    lv: LoopVar
    kind: LoopKind
    exec: LoopExec

    def __repr__(self):
        return f"{self.lv.name}:{self.kind}({self.lv.extent})"

@dataclass
class Schedule:
    loops: list[LoopSched]
    splits: list[SplitLoop]

class SchedulerFunc:
    def __init__(self, efunc):
        spatials = [LoopSched(LoopVar(lv.extent, lv.name), LoopKind.Spatial, Sequential()) for lv in efunc.out_loops]
        reduces = [LoopSched(LoopVar(lv.extent, lv.name), LoopKind.Reduce, Sequential()) for lv in efunc.reduced_vars()]
        self.loops = spatials + reduces
        self.splits = []

    def split(self, lv, outer, inner, factor):
        assert lv.extent % factor == 0, "invalid split factor" # TailStrategy is not implemented yet
        outer.extent = int(lv.extent / factor)
        outer.name = f"{lv.name}__o"

        inner.extent = factor
        inner.name = f"{lv.name}__i"

        self.splits.append(SplitLoop(lv, outer, inner, factor))
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

    def gpu_threads(self, lv, index):
        self._find(lv).exec = GPUThread(index)
        return self

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

        # baseline implementation
        # The outermost spatial loop: parallelize
        # The innermost spatial loop: vectorize
        if spatials: sf.parallel(spatials[0])
        if 1 < len(spatials): sf.vectorize(spatials[-1])

    @staticmethod
    def schedule_gpu(sf):
        spatials = sf.spatial_loops()
        if not spatials: return

        # baseline implementation
        # The outermost spatial loop: block parallel
        # The 2nd outermost spatial loop: thread parallel
        if 1 < len(spatials):
            sf.gpu_blocks(spatials[0], "x").gpu_threads(spatials[1], "x")
            return

        # If there's only one spatial loop, tile it into two and apply block/thread
        io = LoopVar()
        ii = LoopVar()
        sf.split(spatials[0], io, ii, spatials[0].extent / 2)
        sf.gpu_blocks(io, "x").gpu_threads(ii, "x")

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
