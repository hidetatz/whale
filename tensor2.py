# a = Tensor([[1, 2, 3], [1, 2, 3]])
# b = a.sum(axis=0)
# c = Tensor([4, 5, 6])
# d = b + c
# e = 10
# f = d * e
# f.tolist()

# out = ((Tensor([4, 5, 6]) + Tensor([[1, 2, 3], [1, 2, 3]]).sum(axis=0)) * 10).tolist()

# Tensor(Mul, inputs=[
# 	Tensor(Add, inputs=[
# 		Tensor(Const, [4, 5, 6]),
# 		Tensor(Sum, axis=0, inputs=[
# 			Tensor(Const, [[1, 2, 3], [1, 2, 3]]),
# 		]),
# 	]),
# 	Tensor(Const, 10)
# ])
# 
# Tensor(
#   creator=Mul(
#     input=[d, e]
#   )
# )
import weakref
from enum import IntEnum, auto

class CPUBuffer:
	def __init__(self, val):
		self.val = val

class Func:
	def __init__(self, op):
		self.op = op

	def forward(self, inputs, **kwargs):
		self.inputs = inputs
		self.kwargs = kwargs
		f = getattr(self, f"_{self.op}_forward")
		output = f()
		self.output = output
		return output

	def backward(self, grad):
		f = getattr(self, f"_{self.op}_backward")
		return f(grad)

	def _binary_forward(self):
        return Tensor.from_func(shape=self.inputs[0].shape, creator=self)

	def _add_forward(self):
		return self._binary_forward()

    def _add_backward(self, grad):
    	return grad, grad

class Tensor:
	def __init__(self, shape=None, strides=None, creator=None, cpu_buffer=None, grad=None):
		self.shape = shape
		self.strides = strides
		self.cpu_buffer = cpu_buffer
		self.creator = creator
		self.grad = grad

	@classmethod
	def from_const(self, shape, strides, val):
		return Tensor(shape=shape, strides=strides, cpu_buffer=CPUBuffer(val))

	@classmethod
	def from_func(self, shape, creator):
		return Tensor(shape=shape, creator=creator)

	def backward(self):
		if self.grad is None: self.grad = ones_like(self)
        funcs = []
        seen = set()

        def dfs(t):
            if t.creator is None or t in seen: return
            seen.add(t)
            for i in t.creator.inputs: dfs(i)
            funcs.append(t.creator)

        dfs(self)
        funcs.reverse()

        for f in funcs:
            gxs = f.backward(f.output.grad)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else x.grad + gx

    def _compute(self):
    	compiler.compile_and_exec(self)

	def tolist(self):
		if not self.materialized:
			self._compute()

		if self.cpu_buffer is None:
			self._to_cpu()

		return self.cpu_buffer.val

	def __add__(self, r):
		return Func("add").forward((self, r))

#
# factories
# 

def _with_shape(shape, vals):
    strides = tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])
	return Tensor.from_const(shape, strides, vals)

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
            raise ValueError(f"array must be homogeneous: {data}")

        for elem in d:
            f(elem, dim + 1)

    f(val, 0)
    return _with_shape(shape, flattened)

def arange(stop):
	return array([i for i in range(stop)])

def full(shape, val):
	length = math.prod(shape)
	return _with_shape(shape, [val] * length)

def full_like(t, val):
	return full(t.shape, val)

def ones_like(t):
	return full_like(t, 1)

# # dead node elimination
# # constant folding
# # shape inference
# # fusion

# GraphIR(
# 	nodes=[
# 		Node(id=0, op=Const, vals=[[1, 2, 3], [1, 2, 3]], out_shape=[2, 3]),
# 		Node(id=1, op=Const, vals=[4, 5, 6],              out_shape=[3]),
# 		Node(id=2, op=Const, vals=10,                     out_shape=[]),
# 		Node(id=3, op=Sum, inputs=[0],    args={axis:0},  out_shape=[3]),
# 		Node(id=4, op=Add, inputs=[1, 3], args={},        out_shape=[3]),
# 		Node(id=5, op=Mul, inputs=[4, 2], args={},        out_shape=[3]),
# 	],
# 	fused_nodes=[0, 1, 2, 3, Fuse(4, 5)],
# )

# LoopIR(
# 	kernels=[
# 		Kernel(
# 			name="sum_axis0",
# 			index_vars=[IndexVar("idx", 3)],
# 			reduce_vars=[IndexVar("ridx0", 2)],
# 			out_shape=[3],
# 			body=ReduceOp(
# 				op="sum",
# 				reduce_vars=[IndexVar(1, 2)],
# 				body=Load(0, indices=["ridx0", "idx"]),
# 			),
# 			input_nodes=[0],
# 		),
# 		Kernel(
# 			name="fused_add_mul",
# 			index_vars=[IndexVar("idx", 3)],
# 			reduce_vars=[],
# 			out_shape=[3],
# 			body=BinaryOp(
# 				op="mul",
# 				left=BinaryOp(
# 					op="add",
# 					left=Load(1, indices=["idx"]),
# 					right=Load(3, indices=["idx"]),
# 				),
# 				right=Load(2, indices=[]),
# 			),
# 			input_nodes=[1, 3, 2],
# 		),
# 	],
# )

# BackendIR(
# 	programs=[
# 		"void sum_axis0()...",
# 		"void fused_add_mul()...",
# 	]
# )

# # shape: 2, 3, 4
# a = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
# buf = [0, 0, 0]

# # a.sum(axis=[0, 2])  # out_shape=[3]

# for i in range(3):
# 	buf[i] = 0
# 	for j in range(2):
# 		for k in range(4):
# 			buf[i] += a[j][i][k]
# print(buf)
