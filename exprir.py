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
    loopvar: LoopVar
    def inputs(self): return []

@dataclass(eq=False)
class ConstExpr:
    val: int
    def inputs(self): return []

@dataclass(eq=False)
class BinaryExpr:
    op: Ops
    l_expr: Expr
    r_expr: Expr
    def inputs(self): return [self.l_expr, self.r_expr]

@dataclass(eq=False)
class UnaryExpr:
    op: Ops
    expr: Expr
    def inputs(self): return [self.expr]

@dataclass(eq=False)
class ReduceExpr:
    op: Ops
    expr: Expr
    reduced: list[LoopVar]
    def inputs(self): return [self.expr]

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
            if isinstance(e, BinaryExpr): return collect(e.l_expr) + collect(e.r_expr)
            if isinstance(e, UnaryExpr): return collect(e.expr)
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
            elif isinstance(e, BinaryExpr): walk(e.l_expr); walk(e.r_expr)
            elif isinstance(e, UnaryExpr): walk(e.expr)
            elif isinstance(e, ReduceExpr): walk(e.expr)
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
            e = BufferExpr(node=a.node, indices=[IndexExpr(l) for l in out_loops])

        elif a.ctx.op.is_view():
            if a.ctx.op == Ops.Broadcast:
                srcshape = a.ctx.inputs[0].shape
                dstshape = a.shape
                # read broadcasted dimension as 0
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
                l_expr=FuncExpr(func=inputs[0], indices=[IndexExpr(l) for l in out_loops]),
                r_expr=FuncExpr(func=inputs[1], indices=[IndexExpr(l) for l in out_loops]),
            )

        elif a.ctx.op.is_unary():
            e = UnaryExpr(
                op=a.ctx.op,
                expr=FuncExpr(func=inputs[0], indices=[IndexExpr(l) for l in out_loops]),
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
                expr=FuncExpr(func=inputs[0], indices=input_indices),
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
    # fuse Funcs
    #

    def has_reduce(e):
        if isinstance(e, ReduceExpr): return True
        if isinstance(e, BinaryExpr): return has_reduce(e.l_expr) or has_reduce(e.r_expr)
        if isinstance(e, UnaryExpr): return has_reduce(e.expr)
        return False # FuncExpr, BufferExpr, IndexExpr, ConstExpr

    def replace_index(e, index_replace) -> Expr:
        if isinstance(e, IndexExpr): return index_replace[e.loopvar]
        if isinstance(e, BinaryExpr): return BinaryExpr(e.op, replace_index(e.l_expr, index_replace), replace_index(e.r_expr, index_replace))
        if isinstance(e, UnaryExpr): return UnaryExpr(e.op, replace_index(e.expr, index_replace))
        if isinstance(e, FuncExpr): return FuncExpr(e.func, [replace_index(i, index_replace) for i in e.indices])
        if isinstance(e, BufferExpr): return BufferExpr(e.node, [replace_index(i, index_replace) for i in e.indices])
        if isinstance(e, ReduceExpr): return ReduceExpr(e.op, replace_index(e.expr, index_replace), e.reduced)
        return e # ConstExpr

    # count Func reference count
    def count_refs(e, rc):
        if isinstance(e, FuncExpr):
            rc[e.func] = rc.get(e.func, 0) + 1
            if rc[e.func] == 1: count_refs(e.func.expr, rc)
        else:
            for inp in e.inputs(): count_refs(inp, rc)

    refcount = {}
    count_refs(f.expr, refcount)

    def try_fuse(e):
        match e:
            case ConstExpr() | IndexExpr() | BufferExpr(): return e # no parents to fuse, just return
            # for unary, binary, reduce, they are not Func so try to fuse their parents
            case UnaryExpr(): return UnaryExpr(op=e.op, expr=try_fuse(e.expr))
            case BinaryExpr(): return BinaryExpr(op=e.op, l_expr=try_fuse(e.l_expr), r_expr=try_fuse(e.r_expr))
            case ReduceExpr(): return ReduceExpr(op=e.op, expr=try_fuse(e.expr), reduced=e.reduced)
            case FuncExpr():
                is_buffer = isinstance(e.func.expr, BufferExpr) # buffer reference is always fusable
                contains_reduce = has_reduce(e.func.expr)
                referenced_only1 = refcount[e.func] == 1
                fusable = is_buffer or (not contains_reduce and referenced_only1)

                if not fusable:
                    e.func.expr = try_fuse(e.func.expr)
                    return FuncExpr(func=e.func, indices=e.indices)

                # fuse is achieved to replace a fusable FuncExpr with the FuncExpr.func.expr. This removes the kernel boundary which is expressed by
                # Func instance.
                # On the replacement, the FuncExpr.func.expr indices must also be replaced with FuncExpr.func.out_loops indices.
                # index_replace = {LoopVar from original func (which should be replaced from): Index expr from internal expr to be fused (replaced to)}
                index_replace = {loopvar: idxexpr for loopvar, idxexpr in zip(e.func.out_loops, e.indices)}
                # replace the indices in FuncExpr.expr with outer Func out_loops
                fused_expr = replace_index(e.func.expr, index_replace)
                return try_fuse(fused_expr)

            case _: raise RuntimeError(f"unhandled expr: {type(e)}")

    f = Func(out_loops=f.out_loops, out_shape=f.out_shape, out_dtype=f.out_dtype, expr=try_fuse(f.expr), out_buffer=arr.buffer)

    funcs = []
    seen = set()
    def dfs(_f):
        if _f in seen: return
        seen.add(_f)
        for dep in _f.dependent_funcs(): dfs(dep)
        funcs.append(_f)
    dfs(f)

    return funcs
