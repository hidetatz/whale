import math
from dataclasses import dataclass
from enum import IntEnum, auto

import exprir
import backend

class LoopKind(IntEnum):
    Spatial = auto()
    Reduce = auto()

    def __repr__(self): return self.name
    def __str__(self): return self.name

@dataclass
class LoopSched:
    name: str
    extent: int
    kind: LoopKind

    # cpu
    parallel: bool | None = None
    vectorize: bool | None = None
    tile: int | None = None
    unroll: int | None = None

    # gpu
    gpu_blocks: int | None = None
    gpu_threads: int | None = None

    def __repr__(self):
        return f"{self.name}:{self.kind}({self.extent})"

@dataclass
class Schedule:
    scheds: list[LoopSched]

class AutoScheduler:
    @staticmethod
    def schedule_cpu(f):
        spatials = [LoopSched(lv.name, lv.extent, LoopKind.Spatial) for lv in f.out_indices]
        reduces = [LoopSched(lv.name, lv.extent, LoopKind.Reduce) for lv in f.reduced_vars()]

        # baseline implementation
        # The outermost spatial loop: parallelize
        # The innermost spatial loop: vectorize
        if spatials: spatials[0].parallel = True
        if 1 < len(spatials): spatials[-1].vectorize = True
        return Schedule(spatials + reduces)

    @staticmethod
    def schedule_gpu(f):
        spatials = [LoopSched(lv.name, lv.extent, LoopKind.Spatial) for lv in f.out_indices]
        reduces = [LoopSched(lv.name, lv.extent, LoopKind.Reduce) for lv in f.reduced_vars()]

        # baseline implementation
        # The outermost spatial loop: block parallel
        # The 2nd outermost spatial loop: thread parallel
        if 1 < len(spatials):
            spatials[0].gpu_blocks = True
            spatials[1].gpu_threads = True
            return Schedule(spatials + reduces)

        # If there's only one spatial loop, tile it into two and apply block/thread
        threads = 256
        s = spatials[0]
        tiled_spatials = [
            LoopSched(f"{s.name}_outer", math.ceil(s.extent / threads), LoopKind.Spatial, gpu_blocks=True),
            LoopSched(f"{s.name}_inner", threads, LoopKind.Spatial, gpu_threads=True),
        ]
        return Schedule(tiled_spatials + reduces)

def schedule(funcs, scheduler=AutoScheduler):
    return [scheduler.schedule_gpu(func) if backend.gpu_enabled() else scheduler.schedule_cpu(func) for func in funcs]
