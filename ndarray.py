import math
import weakref

from buffer import CPUBuff
import backend
import exprir
import node
import sched
import util
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

    # actual calculation

    def _elemwise_forward(self):
        i = self.inputs[0]
        return ndarray(val=None, dtype=i.dtype, shape=i.shape, strides=None, offset=None, ctx=self)

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
        return ndarray(val=None, dtype=inp.dtype, shape=tuple(newshape), strides=None, offset=None, ctx=self)

    def _sum_forward(self): return self._reduce_forward()
    def _sum_backward(self, grad): pass # todo

    # view

    def _view_forward(self):
        return ndarray(val=None, dtype=self.inputs[0].dtype, shape=self.attrs["shape"], strides=None, offset=None, ctx=self)

    def _reshape_forward(self): return self._view_forward()
    def _reshape_backward(self, grad): return grad.reshape(*self.inputs[0].shape)

    def _broadcast_forward(self): return self._view_forward()
    def _broadcast_backward(self, grad):
        orig_shape = self.inputs[0].shape
        added_axis = tuple(range(grad.ndim - len(orig_shape)))
        expanded_axis = tuple([i + len(added_axis) for i, s in enumerate(orig_shape) if s == 1])
        y = grad.sum(axis=added_axis + expanded_axis, keepdims=True)
        return y if not added_axis else y.reshape(*[s for i, s in enumerate(y.shape) if i not in added_axis])

    def _transpose_forward(self): return self._view_forward()
    def _transpose_backward(self, grad):
        # argsort https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
        perm = sorted(range(len(self.attrs["axes"])), key=self.attrs["axes"].__getitem__)
        return grad.transpose(*perm)

class ndarray:
    def __init__(self, val, dtype, shape, strides, offset, ctx):
        self.node = node.Node(val, dtype, shape, strides, offset, ctx)
        self.grad = None

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
    def buffer(self): return self.node.buffer

    @property
    def ndim(self): return len(self.shape)

    @classmethod
    def wrap(cls, v): return v if isinstance(v, ndarray) else array(v)

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
        return Func(Ops.Reshape).forward((self,), shape=shape)

    def transpose(self, *axes):
        if sorted(axes) != list(range(self.ndim)): raise RuntimeError(f"transapose axes must be wrong: {axes}")
        newshape=[self.shape[a] for a in axes]
        return Func(Ops.Transpose).forward((self,), axes=axes, shape=newshape)

    @property
    def T(self):
        if self.ndim <= 1: return self
        return self.transpose(*list(range(self.ndim))[::-1])

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

    def materialize(self):
        funcs = exprir.convert(self)
        scheds = sched.schedule(funcs, backend.is_gpu())
        backend.codegen_and_exec(funcs, scheds)

    def tolist(self):
        if self.buffer.cpu is None:
            self.buffer.cpu = CPUBuff(backend.to_cpu(self.buffer))
        return self.buffer.cpu.val

    def __binary(self, r, f):
        l, r = self.broadcasted(ndarray.wrap(r))
        return Func(f).forward((l, r))

    def __unary(self, f): return Func(f).forward((self,))
    def __reduce(self, f, axis, keepdims):
        if isinstance(axis, int): axis = (axis,)
        if not axis: axis = (list(range(self.ndim)))
        axis = [a % self.ndim for a in axis]
        return Func(f).forward((self,), axis=axis, keepdims=keepdims)

    def __add__(self, r): return self.__binary(r, Ops.Add)
    def __sub__(self, r): return self.__binary(r, Ops.Sub)
    def __mul__(self, r): return self.__binary(r, Ops.Mul)
    def __truediv__(self, r): return self.__binary(r, Ops.Truediv)
    def __pow__(self, r): return self.__binary(r, Ops.Pow)

    def __neg__(self): return self.__unary(Ops.Neg)

    def sum(self, axis=None, keepdims=False): return self.__reduce(Ops.Sum, axis, keepdims)

    def debug_oneline(self):
        return f"ndarray({self.ctx.op.name if self.ctx else 'Input'} {self.shape}_{self.strides}_{self.offset} cpubuff:{'o' if self.node.buffer.cpu else 'x'} devbuff:{'o' if self.node.buffer.dev else 'x'})"

    def debug(self):
        def f(depth, arr):
            indent = "  " * depth
            trail_comma = "," if depth != 0 else ""

            if not arr.ctx or not arr.ctx.inputs:
                return f"{indent}{arr.debug_oneline()}{trail_comma}"

            inputs = "[\n" + "\n".join([f(depth + 1, i) for i in arr.ctx.inputs]) + f"\n{indent}]"
            return f"{indent}{arr.debug_oneline().rstrip(')')} inputs: {inputs}{trail_comma}"

        return f(0, self)

#
# factories
# 

def _const(shape, val):
    dtype = int64 if val and type(val[0]) is int else float64
    strides = util.strides_from_shape(shape)
    return ndarray(val, dtype, shape, strides, 0, Func(Ops.Const))

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
