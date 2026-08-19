import ndarray

a = ndarray.arange(25).reshape(5, 5)
b = ndarray.where(a > 10, a, a * 10)
b.materialize()
print(b.tolist())
