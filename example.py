import ndarray

a = ndarray.arange(25).reshape(5, 5)
b = a.dilate(1, 2)
b.materialize()
b.backward()
print(b.tolist())
a.grad.materialize()
print(a.grad.tolist())
