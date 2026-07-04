import math

def strides_from_shape(shape): return tuple([math.prod(shape[i + 1 :]) for i in range(len(shape))])

