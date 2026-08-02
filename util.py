import math
import typing

def strides_from_shape(shape): return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])
def strjoin(sep: str, arr: list[typing.Any]): return sep.join(a.__str__() for a in arr)

