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
    def parents(self): return self.inputs()
    def print_oneline(self): return f"IndexExpr({self.idx})"

@dataclass(eq=False)
class ConstExpr:
    val: int
    def inputs(self): return []
    def parents(self): return self.inputs()
    def print_oneline(self): return f"ConstExpr({self.val})"

@dataclass(eq=False)
class BinaryExpr:
    op: Ops
    left: Expr
    right: Expr
    def inputs(self): return [self.left, self.right]
    def parents(self): return self.inputs()
    def print_oneline(self): return f"BinaryExpr({self.op})"

@dataclass(eq=False)
class UnaryExpr:
    op: Ops
    operand: Expr
    def inputs(self): return [self.operand]
    def parents(self): return self.inputs()
    def print_oneline(self): return f"UnaryExpr({self.op})"

@dataclass(eq=False)
class ReduceExpr:
    op: Ops
    operand: Expr
    reduced: list[LoopVar]
    def inputs(self): return [self.operand]
    def parents(self): return self.inputs()
    def print_oneline(self): return f"ReduceExpr({self.op}:{self.reduced})"

@dataclass(eq=False)
class FuncExpr:
    src: Func
    indices: list[Expr]
    def inputs(self): return [self.src]
    def parents(self): return self.inputs()
    def print_oneline(self): return f"FuncExpr()"

@dataclass(eq=False)
class BufferExpr:
    src: Node
    indices: list[Expr]
    def inputs(self): return []
    def parents(self): return self.inputs()
    def print_oneline(self): return f"BufferExpr(indices={self.indices})"

Expr = IndexExpr | ConstExpr | BinaryExpr | UnaryExpr | ReduceExpr | FuncExpr | BufferExpr

@dataclass(eq=False)
class Func:
    out_indices: list[LoopVar]
    out_shape: list[int]
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
                if e.src not in seen:
                    seen.add(e.src)
                    bufs.append(e)
            elif isinstance(e, FuncExpr):
                if e.src not in seen:
                    seen.add(e.src)
                    fncs.append(e)
            elif isinstance(e, BinaryExpr): walk(e.left); walk(e.right)
            elif isinstance(e, UnaryExpr): walk(e.operand)
            elif isinstance(e, ReduceExpr): walk(e.operand)
        walk(self.expr)
        return bufs, fncs

    def parents(self): return [self.expr]
    def print_oneline(self): return f"Func({self.out_indices}_{self.out_shape})"

    def __repr__(self):
        return repr_tree(self, parent_str="expr")

def repr_tree(tree, indent="  ", parent_str="parents"):
    def f(depth, t):
        indentstr = indent * depth
        trail_comma = "," if depth != 0 else ""

        parents = t.parents()
        if not parents:
            return f"{indentstr}{t.print_oneline()}{trail_comma}"

        inputs = "[\n" + "\n".join([f(depth + 1, p) for p in parents]) + f"\n{indentstr}]"
        return f"{indentstr}{t.print_oneline().rstrip(')')} {parent_str}: {inputs}{trail_comma}"

    return f(0, tree)

@dataclass
class ExprIR:
    funcs: list[Func] # topo-sorted

def convert(arr):
    #
    # convert ndarray into Func one by one
    # 

    def _lower(a, cache):
        def loopvar_name(i, prefix=""):
            names = "ijklmnpq"
            idx = names[i] if i < len(names) else f"v{i}"
            return f"{prefix}{idx}"

        indices = [LoopVar(s, loopvar_name(i)) for i, s in enumerate(a.shape)]

        if a.ctx.op.is_const():
            return Func(
                out_indices=indices,
                out_shape=a.shape,
                out_dtype=a.dtype,
                expr=BufferExpr(src=a.node, indices=[IndexExpr(idx) for idx in indices]),
                out_buffer=a.buffer,
            )

        inputs = [lower(inp, cache) for inp in a.ctx.inputs]
        
        if a.ctx.op.is_view():
            src = inputs[0]
            if a.ctx.op == Ops.Broadcast:
                srcshape = a.ctx.inputs[0].shape
                dstshape = a.shape
                # ignore padded dim, replace originally-1 dim with 0
                new_view_indices = [ConstExpr(0) if s == 1 else IndexExpr(d) for s, d in zip(srcshape, indices[len(dstshape) - len(srcshape):])]
            elif a.ctx.op == Ops.Transpose:
                # axes is a map from output axis -> input axis.
                # to calculate index, input -> output map is needed, so inverts
                axes = a.ctx.attrs["axes"]
                inv = [0] * len(axes)
                for i, ax in enumerate(axes): inv[ax] = i
                new_view_indices = [IndexExpr(indices[inv[i]]) for i in range(len(axes))]
            else:
                raise RuntimeError(f"not implemented op: {a.ctx.op.name}")

            return Func(out_indices=indices, out_shape=a.shape, out_dtype=a.dtype, expr=FuncExpr(src=src, indices=new_view_indices), out_buffer=a.buffer)
                

        if a.ctx.op.is_binary():
            return Func(
                out_indices=indices,
                out_shape=a.shape,
                out_dtype=a.dtype,
                expr=BinaryExpr(
                    op=a.ctx.op,
                    left=FuncExpr(src=inputs[0], indices=[IndexExpr(idx) for idx in indices]),
                    right=FuncExpr(src=inputs[1], indices=[IndexExpr(idx) for idx in indices]),
                ),
                out_buffer=a.buffer,
            )

        if a.ctx.op.is_unary():
            return Func(
                out_indices=indices,
                out_shape=a.shape,
                out_dtype=a.dtype,
                expr=UnaryExpr(
                    op=a.ctx.op,
                    operand=FuncExpr(src=inputs[0], indices=[IndexExpr(idx) for idx in indices]),
                ),
                out_buffer=a.buffer,
            )

        if a.ctx.op.is_reduce():
            axis = a.ctx.attrs["axis"]
            keepdims = a.ctx.attrs["keepdims"]
            # create LoopVars for reduced axis {axis: LoopVar(size for the axis)}
            reduced = {ax: LoopVar(a.ctx.inputs[0].shape[ax], loopvar_name(i, prefix="r")) for i, ax in enumerate(sorted(axis))}

            # pick up the non-reduced axis loopvars from out_indices.
            # if not keepdims, the indices only contains the spatial indices.
            spatial_loopvars = iter(lv for i, lv in enumerate(indices) if i not in axis) if keepdims else iter(indices)
            # create input indices from pre-created loopvars.
            input_indices = [IndexExpr(reduced[dim] if dim in axis else next(spatial_loopvars)) for dim in range(a.ctx.inputs[0].ndim)]

            return Func(
                out_indices=indices,
                out_shape=a.shape,
                out_dtype=a.dtype,
                expr=ReduceExpr(
                    op=a.ctx.op,
                    operand=FuncExpr(src=inputs[0], indices=input_indices),
                    reduced=list(reduced.values()),
                ),
                out_buffer=a.buffer,
            )

        raise RuntimeError(f"not implemented op: {a.ctx.op.name}")

    def lower(a, cache):
        # this is needed for the ndarray referenced more than once
        if a in cache: return cache[a]
        f = _lower(a, cache)
        cache[a] = f
        return f

    cache = {}
    f = lower(arr, cache)

    #
    # fuse Funcs if possible
    # 

    def count_refs(e, refcount):
        if isinstance(e, FuncExpr):
            refcount[e.src] = refcount.get(e.src, 0) + 1
            if refcount[e.src] == 1:
                count_refs(e.src.expr, refcount)
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
        if isinstance(e, FuncExpr): return FuncExpr(e.src, [subst(i, mapping) for i in e.indices])
        if isinstance(e, BufferExpr): return BufferExpr(e.src, [subst(i, mapping) for i in e.indices])
        if isinstance(e, ReduceExpr): return ReduceExpr(e.op, subst(e.operand, mapping), e.reduced)
        return e # ConstExpr

    def inline(e):
        if isinstance(e, FuncExpr):
            # fusable?
            # todo: support epilogue/prologue fusion
            if isinstance(e.src.expr, BufferExpr) or (not has_reduce(e.src.expr) and refcount.get(e.src, 0) == 1):
                assert len(e.src.out_indices) == len(e.indices)
                mp = {oidx: iidx for oidx, iidx in zip(e.src.out_indices, e.indices)}
                return inline(subst(e.src.expr, mp))
            else:
                e.src.expr = inline(e.src.expr)
                return FuncExpr(src=e.src, indices=e.indices)
        if isinstance(e, BinaryExpr): return BinaryExpr(op=e.op, left=inline(e.left), right=inline(e.right))
        if isinstance(e, UnaryExpr): return UnaryExpr(op=e.op, operand=inline(e.operand))
        if isinstance(e, ReduceExpr): return ReduceExpr(op=e.op, operand=inline(e.operand), reduced=e.reduced)
        return e # ConstExpr, IndexExpr, BufferExpr

    f = Func(out_indices=f.out_indices, out_shape=f.out_shape, out_dtype=f.out_dtype, expr=inline(f.expr), out_buffer=arr.buffer)

    funcs = []
    seen = set()
    def dfs(_f):
        if _f in seen: return
        seen.add(_f)
        for dep in _f.dependent_funcs(): dfs(dep)
        funcs.append(_f)
    dfs(f)

    return ExprIR(funcs)

if __name__ == "__main__":
    from ndarray import array
    a = array([[4, 5, 6], [4, 5, 6]])  # 2, 3
    b = a.sum(axis=0)
    c = array([4, 5, 6])  # 2, 3
    d = array([4, 5, 6])  # 2, 3
    e = b * c + d
    print(e.debug())
    eir = convert(e)
