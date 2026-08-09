import math
import typing

def strides_from_shape(shape: tuple[int, ...]): return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])
def strjoin(sep: str, arr: list[typing.Any]): return sep.join(a.__str__() for a in arr)
def argsort(seq: list[typing.Any]): return sorted(range(len(seq)), key=seq.__getitem__)
