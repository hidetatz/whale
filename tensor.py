import math
from enum import Enum, auto

class Calculation:
    def forward(self, input): raise NotImplementedError()
    def backward(self, grad): raise NotImplementedError()

class Add(Calculation):
    def forward(self, input):
        if len(input) != 2:
            raise ValueError(f"wrong count of operand passed to Add(): {input}")
        return Tensor(shape=input[0].shape, stride=input[0].stride, op=TensorOp.ADD, input=input, dtype=input[0].dtype)

    def backward(self, grad):
        return [grad, grad]

class DType(Enum):
    I32 = auto()
    F32 = auto()

    @classmethod
    def from_type(cls, typ):
        if typ == int:
            return cls.I32
        if typ == float:
            return cls.F32

class TensorOp(Enum):
    CONST = auto()

    # arithmetic
    ADD = auto()

class Tensor:
    def __init__(self, data=[], shape=[], stride=[], op=None, creator=None, input=[], materialized=False, dtype=None):
        self.data = data
        self.shape = shape
        self.stride = stride
        self.op = op
        self.creator = creator
        self.input = input
        self.materialized = materialized
        self.dtype = dtype

    @classmethod
    def from_calc(cls, calc, input):
        c = calc()
        t = c.forward(input)
        t.creator = c
        return t

    def __add__(self, r):
        return Tensor.from_calc(Add, [self, r])

    def __str__(self):
        if self.materialized:
            if len(self.shape) == 0:
                return f"Tensor({self.data[0]})"

            if len(self.shape) == 1:
                return f"Tensor({self.data})"

            # todo: implement
            return f"Tensor({self.data}), {self.shape}"

        else:
            if self.op == TensorOp.ADD:
                return f"{self.input[0]} + {self.input[1]}"

            return "unknown"
        
    def debug(self):
        return f"data: {self.data}, shape: {self.shape}, stride: {self.stride}, op: {self.op}, materialized: {self.materialized}, dtype: {self.dtype}"

def tensor(arr):
    if type(arr) == int or type(arr) == float:
        return Tensor(data=[arr], op=TensorOp.CONST, materialized=True, dtype=DType.from_type(type(arr)))

    elif type(arr) == list:
        data = []
        shape = []
        def f(d, dim):
            if type(d) != int and type(d) != float and type(d) != list:
                raise ValueError(f"array must be a multi-dimensional array of int or float: {arr}")

            if type(d) == int or type(d) == float:
                data.append(d)
                return

            # d must be list here
            length = len(d)
            if len(shape) == dim:
                shape.append(length)
            else:
                if length != shape[dim]:
                    raise ValueError(f"array must be homogeneous: {arr}")

            for elem in d:
                f(elem, dim+1)

        f(arr, 0)
        stride = [math.prod(shape[i+1:]) for i in range(len(shape))]
        return Tensor(data=data, shape=shape, stride=stride, op=TensorOp.CONST, materialized=True, dtype=DType.from_type(type(data[0])))


print(array(1).debug())
print(array([1, 2, 3]).debug())
print(array([[1, 2], [3, 4]]).debug())

t1 = tensor(1)
t2 = tensor(2)
print(t1, t2)
print(t1+t2)
