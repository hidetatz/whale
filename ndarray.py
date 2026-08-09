import math
import weakref
from functools import reduce
from operator import mul

import backend
import materialize
import util
from buffer import CPUBuff
from node import Node
from ops import Ops
from dtype import int32, int64, float32, float64

class Func:
    def __init__(self, op):
        self.op = op
        self.inputs = []
        self.attrs = {}
        self.output = None

    # forward and backward path

    def forward(self, inputs, **kwargs):
        self.inputs = inputs
        self.attrs = kwargs
        f = getattr(self, f"_{self.op.name.lower()}_forward")
        output = f()
        self.output = weakref.ref(output)
        return output

    def backward(self, grad):
        f = getattr(self, f"_{self.op.name.lower()}_backward")
        out = f(grad)
        return out if isinstance(out, tuple) else (out,)

    @property
    def input(self): return self.inputs[0]

    # actual calculation

    def _elemwise_forward(self):
        i = self.input
        return ndarray._from_prim(val=None, dtype=i.dtype, shape=i.shape, strides=util.strides_from_shape(i.shape), offset=0, ctx=self)

    # unary

    def _neg_forward(self): return self._elemwise_forward()
    def _neg_backward(self, grad): return -grad

    # binary

    def _add_forward(self): return self._elemwise_forward()
    def _add_backward(self, grad): return grad, grad

    def _sub_forward(self): return self._elemwise_forward()
    def _sub_backward(self, grad): return grad, -grad

    def _mul_forward(self): return self._elemwise_forward()
    def _mul_backward(self, grad): return grad * self.inputs[1], grad * self.inputs[0]

    def _truediv_forward(self): return self._elemwise_forward()
    def _truediv_backward(self, grad): return grad / self.inputs[1], grad * (-self.inputs[0] / self.inputs[1] ** 2)

    def _pow_forward(self): return self._elemwise_forward()
    def _pow_backward(self, grad): return self.inputs[1] * self.inputs[0] ** (self.inputs[1] - 1) * grad

    # reduce

    def _reduce_forward(self):
        inp = self.inputs[0]
        axis = self.attrs["axis"]
        kd = self.attrs["keepdims"]
        if kd: newshape = [1 if i in axis else s for i, s in enumerate(inp.shape)]
        else: newshape = [s for i, s in enumerate(inp.shape) if i not in axis]
        return ndarray._from_prim(val=None, dtype=inp.dtype, shape=tuple(newshape), strides=util.strides_from_shape(newshape), offset=0, ctx=self)

    def _sum_forward(self): return self._reduce_forward()
    def _sum_backward(self, grad): pass # todo

    # view

    def _reshape_forward(self):
        inp = self.inputs[0]
        target_shape = self.attrs["shape"]
        new_node = Node(dtype=inp.dtype, shape=target_shape, strides=util.strides_from_shape(target_shape), offset=0, ctx=self)
        return ndarray(new_node)
    def _reshape_backward(self, grad): return grad.reshape(*self.inputs[0].shape)

    def _broadcast_forward(self):
        inp = self.inputs[0]
        target_shape = self.attrs["shape"]
        ndim_diff = len(target_shape) - len(inp.shape)
        padded_strides = [0] * ndim_diff + list(inp.strides)
        padded_src_shape = [1] * ndim_diff + list(inp.shape)
        new_strides = [0 if s == 1 else st for s, st in zip(padded_src_shape, padded_strides)]
        new_node = Node(dtype=inp.dtype, shape=target_shape, strides=tuple(new_strides), offset=inp.offset, ctx=self)
        return ndarray(new_node)
    def _broadcast_backward(self, grad):
        orig_shape = self.inputs[0].shape
        added_axis = tuple(range(grad.ndim - len(orig_shape)))
        expanded_axis = tuple([i + len(added_axis) for i, s in enumerate(orig_shape) if s == 1])
        y = grad.sum(axis=added_axis + expanded_axis, keepdims=True)
        return y if not added_axis else y.reshape(*[s for i, s in enumerate(y.shape) if i not in added_axis])

    def _transpose_forward(self):
        axes = self.attrs["axes"]
        new_shape = tuple([self.input.shape[a] for a in axes])
        new_strides = tuple([self.input.strides[a] for a in axes])
        new_node = Node(dtype=self.input.dtype, shape=new_shape, strides=new_strides, offset=self.input.offset, ctx=self)
        return ndarray(new_node)
    def _transpose_backward(self, grad):
        return grad.transpose(*util.argsort(self.attrs["axes"]))

    def _slice_forward(self):
        newoffset, newshape, newstrides = self.input.offset, [], []
        for dim, s in enumerate(list(self.attrs["subscript"]) + [slice(None)] * (self.input.ndim - len(self.attrs["subscript"]))):
            match s:
                case int():
                    if s < 0: s += self.input.shape[dim]
                    newoffset += s * self.input.strides[dim]
                case slice():
                    start = s.start if s.start is not None else 0
                    stop = s.stop if s.stop is not None else self.input.shape[dim]
                    step = s.step if s.step is not None else 1
                    newoffset += start * self.input.strides[dim]
                    newshape.append((stop - start + step - 1) // step)
                    newstrides.append(self.input.strides[dim] * step)
        return ndarray(Node(dtype=self.input.dtype, shape=tuple(newshape), strides=tuple(newstrides), offset=newoffset, ctx=self))
    def _slice_backward(self, grad): pass

    def _contiguous_forward(self):
        return ndarray(Node(dtype=self.input.dtype, shape=self.input.shape, strides=util.strides_from_shape(self.input.shape), offset=0, ctx=self))
    def _contiguous_backward(self): pass

class ndarray:
    def __init__(self, node: Node):
        self.node = node
        self.grad = None

    # creation

    @classmethod
    def _from_prim(cls, val, dtype, shape, strides, offset, ctx):
        return cls(Node(dtype=dtype, shape=shape, strides=strides, offset=offset, val=val, ctx=ctx))

    @classmethod
    def wrap(cls, v): return v if isinstance(v, ndarray) else array(v)

    # properties

    @property
    def dtype(self): return self.node.dtype
    @property
    def shape(self): return self.node.shape
    @property
    def strides(self): return self.node.strides
    @property
    def offset(self): return self.node.offset
    @property
    def ctx(self): return self.node.ctx
    @property
    def buffer(self): return self.base().node.buffer
    @property
    def ndim(self): return len(self.shape)

    def base(self): return self if not self.ctx.op.is_view() else self.ctx.inputs[0].base()

    # unary

    def __unary(self, f): return Func(f).forward((self,))

    def __neg__(self): return self.__unary(Ops.Neg)
    def log(self): return self.__unary(Ops.Log)

    # binary

    def __binary(self, r, f):
        l, r = self.broadcasted(ndarray.wrap(r))
        return Func(f).forward((l, r))

    def __add__(self, r): return self.__binary(r, Ops.Add)
    def __sub__(self, r): return self.__binary(r, Ops.Sub)
    def __mul__(self, r): return self.__binary(r, Ops.Mul)
    def __truediv__(self, r): return self.__binary(r, Ops.Truediv)
    def __pow__(self, r): return self.__binary(r, Ops.Pow)

    # reduce

    def __reduce(self, f, axis, keepdims):
        if isinstance(axis, int): axis = (axis,)
        if not axis: axis = (list(range(self.ndim)))
        axis = [a % self.ndim for a in axis]
        return Func(f).forward((self,), axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False): return self.__reduce(Ops.Sum, axis, keepdims)

    # view and copy

    def __getitem__(self, subscript):
        if not isinstance(subscript, tuple): subscript = (subscript,)
        if self.ndim < len(subscript): raise RuntimeError(f"too many indices for array, array is {self.ndim} dimensional but {len(subscript)} was given")
        for s in subscript:
            if not type(s) is int and not type(s) is slice:  raise RuntimeError(f"only int or slice are valid as index, got {type(s)}")
        return Func(Ops.Slice).forward((self,), subscript=subscript)

    def broadcast_to(self, shape):
        return Func(Ops.Broadcast).forward((self,), shape=shape)

    def broadcasted(self, r):
        # determine the new shape
        ls1 = list(self.shape)
        ls2 = list(r.shape)
        maxlen = max(len(ls1), len(ls2))
        ls1 = [1] * (maxlen - len(ls1)) + ls1
        ls2 = [1] * (maxlen - len(ls2)) + ls2
        newshape = []
        for d1, d2 in zip(ls1, ls2):
            if d1 == d2: newshape.append(d1)
            elif d1 == 1: newshape.append(d2)
            elif d2 == 1: newshape.append(d1)
            else: raise RuntimeError(f"shapes are not broadcastable: {self.shape} and {r.shape}")

        newshape = tuple(newshape)
        l = self
        if l.shape != newshape: l = l.broadcast_to(newshape)
        if r.shape != newshape: r = r.broadcast_to(newshape)
        return l, r

    def reshape(self, *shape):
        if math.prod(shape) != math.prod(self.shape): raise RuntimeError(f"invalid reshape {shape} for size {math.prod(self.shape)}")
        return Func(Ops.Reshape).forward((self.contiguous(),), shape=shape)

    @property
    def T(self):
        if self.ndim <= 1: return self
        return self.transpose(*list(range(self.ndim))[::-1])

    def transpose(self, *axes):
        if sorted(axes) != list(range(self.ndim)): raise RuntimeError(f"transapose axes must be wrong: {axes}")
        return Func(Ops.Transpose).forward((self,), axes=axes)

    def is_contiguous(self): return self.strides == util.strides_from_shape(self.shape)

    def contiguous(self): return self if self.is_contiguous() else Func(Ops.Contiguous).forward((self,))

    # gradient

    def backward(self):
        if self.grad is None: self.grad = ones_like(self)
        funcs = []
        seen = set()

        def dfs(t):
            if t.ctx is None or not t.ctx.inputs or t in seen: return
            seen.add(t)
            for i in t.ctx.inputs: dfs(i)
            funcs.append(t.ctx)

        dfs(self)
        funcs.reverse()

        for f in funcs:
            gxs = f.backward(f.output().grad)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx

    # materialization

    def materialize(self): materialize.materialize(self)

    def tolist(self):
        if self.buffer.cpu is None:
            self.buffer.cpu = CPUBuff(backend.to_cpu(self.buffer))
        return self._to_ndlist()

    def _to_ndlist(self):
        if not self.shape: return self.buffer.cpu.val[self.offset]
        data = self.buffer.cpu.val
        def f(shape, strides, offset):
            if not shape: return data[offset]
            return [f(shape[1:], strides[1:], offset+i*strides[0]) for i in range(shape[0])]
        return f(self.shape, self.strides, self.offset)

    # representation

    def __repr__(self): return str(self)

    def __str__(self):
        if self.buffer is None:
            return f"{self.ctx.op.name if self.ctx else "Input"} shape={self.shape} strides={self.strides} offset={self.offset} dtype={self.dtype} buffer=None"
        if self.buffer.cpu is None:
            return f"{self.ctx.op.name if self.ctx else "Input"} shape={self.shape} strides={self.strides} offset={self.offset} dtype={self.dtype} cpu={self.buffer.cpu} dev={self.buffer.dev}"
        else:
            return str(self._to_ndlist())

    def inputs(self): return list(self.ctx.inputs) if self.ctx and self.ctx.inputs else []

#
# factories
# 

def _const(shape, val):
    dtype = int64 if val and type(val[0]) is int else float64
    strides = util.strides_from_shape(shape)
    return ndarray._from_prim(val, dtype, shape, strides, 0, Func(Ops.Const))

def array(val):
    flattened = []
    shape = []

    def f(d, dim):
        if isinstance(d, int) or isinstance(d, float):
            flattened.append(d)
            return

        # d must be list here
        length = len(d)
        if len(shape) == dim:
            shape.append(length)
        elif length != shape[dim]:
            raise ValueError(f"array must be homogeneous: {val}")

        for elem in d:
            f(elem, dim + 1)

    f(val, 0)
    return _const(tuple(shape), flattened)

def arange(stop):
    return array([i for i in range(stop)])

def full(shape, val):
    return _const(shape, [val] * math.prod(shape))

def full_like(t, val):
    return full(t.shape, val)

def ones_like(t):
    return full_like(t, 1)

def zeros_like(t):
    return full_like(t, 0)
