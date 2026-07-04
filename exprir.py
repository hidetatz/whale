from dataclasses import dataclass
from node import Node

from buffer import Buffer
from dtype import DType, int64
from ops import Ops

@dataclass
class LoopVar:
    extent: int
    name: str = "" # for debug
    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    def __repr__(self): return f"{self.name}({self.extent})" if self.name else f"LoopVar{self.extent}"

#
# Expressions
#

@dataclass(eq=False)
class IndexExpr:
    idx: LoopVar
    def inputs(self): return []

@dataclass(eq=False)
class ConstExpr:
    val: int
    def inputs(self): return []

@dataclass(eq=False)
class BinaryExpr:
    op: Ops
    left: Expr
    right: Expr
    def inputs(self): return [self.left, self.right]

@dataclass(eq=False)
class UnaryExpr:
    op: Ops
    operand: Expr
    def inputs(self): return [self.operand]

@dataclass(eq=False)
class ReduceExpr:
    op: Ops
    operand: Expr
    reduced: list[LoopVar]
    def inputs(self): return [self.operand]

@dataclass(eq=False)
class FuncExpr:
    func: Func
    indices: list[Expr]
    def inputs(self): return [self.func]

@dataclass(eq=False)
class BufferExpr:
    node: Node
    indices: list[Expr]
    def inputs(self): return []

Expr = IndexExpr | ConstExpr | BinaryExpr | UnaryExpr | ReduceExpr | FuncExpr | BufferExpr

@dataclass(eq=False)
class Func:
    out_loops: list[LoopVar]  # loop index to form this func result
    out_shape: list[int] # shape of this func result
    out_dtype: DType
    expr: Expr
    out_buffer: Buffer

    def dependent_funcs(self):
        deps = []
        seen = set()
        def dfs(e):
            for inp in e.inputs():
                if isinstance(inp, Func):
                    if inp not in seen:
                        seen.add(inp)
                        deps.append(inp)
                else:
                    dfs(inp)
        dfs(self.expr)
        return deps

    def reduced_vars(self):
        def collect(e):
            if isinstance(e, ReduceExpr): return e.reduced
            if isinstance(e, BinaryExpr): return collect(e.left) + collect(e.right)
            if isinstance(e, UnaryExpr): return collect(e.operand)
            return []
        return collect(self.expr)

    def inputs(self):
        bufs, fncs = [], []
        seen = set()
        def walk(e):
            if isinstance(e, BufferExpr):
                if e.node not in seen:
                    seen.add(e.node)
                    bufs.append(e)
            elif isinstance(e, FuncExpr):
                if e.func not in seen:
                    seen.add(e.func)
                    fncs.append(e)
            elif isinstance(e, BinaryExpr): walk(e.left); walk(e.right)
            elif isinstance(e, UnaryExpr): walk(e.operand)
            elif isinstance(e, ReduceExpr): walk(e.operand)
        walk(self.expr)
        return bufs, fncs

def convert(arr):
    def loopvar_name(i, prefix=""):
        names = "ijklmnpq"
        idx = names[i] if i < len(names) else f"v{i}"
        return f"{prefix}{idx}"

    # Converts ndarray to Func one by one using dfs.
    # Fusion does not happen here.
    def arr_to_func(a, memo):
        # this is needed for the ndarray referenced more than once
        if a in memo: return memo[a]

        # make out_loops from array shape.
        out_loops = [LoopVar(s, loopvar_name(i)) for i, s in enumerate(a.shape)]

        # convert array inputs into func recursively
        inputs = [arr_to_func(inp, memo) for inp in a.ctx.inputs]

        #
        # make expr from ndarray dependent inputs
        #

        if a.ctx.op.is_const():
            e = BufferExpr(node=a.node, indices=[IndexExpr(idx) for idx in out_loops])

        elif a.ctx.op.is_view():
            if a.ctx.op == Ops.Broadcast:
                srcshape = a.ctx.inputs[0].shape
                dstshape = a.shape
                # ignore padded dim, replace originally-1 dim with 0
                new_view_indices = [ConstExpr(0) if s == 1 else IndexExpr(d) for s, d in zip(srcshape, out_loops[len(dstshape) - len(srcshape):])]
            elif a.ctx.op == Ops.Transpose:
                # axes is a map from output axis -> input axis.
                # to calculate index, input -> output map is needed, so inverts
                axes = a.ctx.attrs["axes"]
                inv = [0] * len(axes)
                for i, ax in enumerate(axes): inv[ax] = i
                new_view_indices = [IndexExpr(out_loops[inv[i]]) for i in range(len(axes))]
            else:
                raise RuntimeError(f"not implemented op: {a.ctx.op.name}")

            e = FuncExpr(func=inputs[0], indices=new_view_indices)
                
        elif a.ctx.op.is_binary():
            e = BinaryExpr(
                op=a.ctx.op,
                left=FuncExpr(func=inputs[0], indices=[IndexExpr(idx) for idx in out_loops]),
                right=FuncExpr(func=inputs[1], indices=[IndexExpr(idx) for idx in out_loops]),
            )

        elif a.ctx.op.is_unary():
            e = UnaryExpr(
                op=a.ctx.op,
                operand=FuncExpr(func=inputs[0], indices=[IndexExpr(idx) for idx in out_loops]),
            )

        elif a.ctx.op.is_reduce():
            axis = a.ctx.attrs["axis"]
            keepdims = a.ctx.attrs["keepdims"]
            # create LoopVars for reduced axis {axis: LoopVar(size for the axis)}
            reduced = {ax: LoopVar(a.ctx.inputs[0].shape[ax], loopvar_name(i, prefix="r")) for i, ax in enumerate(sorted(axis))}

            # pick up the non-reduced axis loopvars from out_loops.
            # if not keepdims, the indices only contains the spatial indices.
            spatial_loopvars = iter(lv for i, lv in enumerate(out_loops) if i not in axis) if keepdims else iter(out_loops)
            # create input indices from pre-created loopvars.
            input_indices = [IndexExpr(reduced[dim] if dim in axis else next(spatial_loopvars)) for dim in range(a.ctx.inputs[0].ndim)]

            e = ReduceExpr(
                op=a.ctx.op,
                operand=FuncExpr(func=inputs[0], indices=input_indices),
                reduced=list(reduced.values()),
            )

        else:
            raise RuntimeError(f"not implemented op: {a.ctx.op.name}")

        f = Func(out_loops=out_loops, out_shape=a.shape, out_dtype=a.dtype, expr=e, out_buffer=a.buffer)
        memo[a] = f
        return f

    # create func tree from arr
    f = arr_to_func(arr, {})

    #
    # fuse Funcs if possible
    # 

    def count_refs(e, refcount):
        if isinstance(e, FuncExpr):
            refcount[e.func] = refcount.get(e.func, 0) + 1
            if refcount[e.func] == 1:
                count_refs(e.func.expr, refcount)
        if isinstance(e, BinaryExpr):
            count_refs(e.left, refcount)
            count_refs(e.right, refcount)
        if isinstance(e, UnaryExpr):
            count_refs(e.operand, refcount)
        if isinstance(e, ReduceExpr):
            count_refs(e.operand, refcount)

    refcount = {}
    count_refs(f.expr, refcount)

    def has_reduce(e):
        if isinstance(e, ReduceExpr): return True
        if isinstance(e, BinaryExpr): return has_reduce(e.left) or has_reduce(e.right)
        if isinstance(e, UnaryExpr): return has_reduce(e.operand)
        return False # FuncExpr, BufferExpr, IndexExpr, ConstExpr

    def subst(e, mapping):
        if isinstance(e, IndexExpr): return mapping.get(e.idx, e)
        if isinstance(e, BinaryExpr): return BinaryExpr(e.op, subst(e.left, mapping), subst(e.right, mapping))
        if isinstance(e, UnaryExpr): return UnaryExpr(e.op, subst(e.operand, mapping))
        if isinstance(e, FuncExpr): return FuncExpr(e.func, [subst(i, mapping) for i in e.indices])
        if isinstance(e, BufferExpr): return BufferExpr(e.node, [subst(i, mapping) for i in e.indices])
        if isinstance(e, ReduceExpr): return ReduceExpr(e.op, subst(e.operand, mapping), e.reduced)
        return e # ConstExpr

    def inline(e):
        if isinstance(e, FuncExpr):
            # fusable?
            # todo: support epilogue/prologue fusion
            if isinstance(e.func.expr, BufferExpr) or (not has_reduce(e.func.expr) and refcount.get(e.func, 0) == 1):
                assert len(e.func.out_loops) == len(e.indices)
                mp = {oidx: iidx for oidx, iidx in zip(e.func.out_loops, e.indices)}
                return inline(subst(e.func.expr, mp))
            else:
                e.func.expr = inline(e.func.expr)
                return FuncExpr(func=e.func, indices=e.indices)
        if isinstance(e, BinaryExpr): return BinaryExpr(op=e.op, left=inline(e.left), right=inline(e.right))
        if isinstance(e, UnaryExpr): return UnaryExpr(op=e.op, operand=inline(e.operand))
        if isinstance(e, ReduceExpr): return ReduceExpr(op=e.op, operand=inline(e.operand), reduced=e.reduced)
        return e # ConstExpr, IndexExpr, BufferExpr

    f = Func(out_loops=f.out_loops, out_shape=f.out_shape, out_dtype=f.out_dtype, expr=inline(f.expr), out_buffer=arr.buffer)

    funcs = []
    seen = set()
    def dfs(_f):
        if _f in seen: return
        seen.add(_f)
        for dep in _f.dependent_funcs(): dfs(dep)
        funcs.append(_f)
    dfs(f)

    return funcs
