import ndarray

a = ndarray.array([[0, 1, 2], [3, 4, 5]])  # (2, 3)
b = ndarray.arange(6).reshape(2, 3)  # (2, 3)
c = a + b
c.backward()

a.grad.materialize()
b.grad.materialize()
print(a.grad.tolist())
print(b.grad.tolist())
