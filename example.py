import ndarray

a = ndarray.arange(25).reshape(5, 5)
b = ndarray.arange(25).reshape(5, 5)
c = a @ b
c.materialize()
print(c.tolist())
