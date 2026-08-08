import ndarray
import backend

a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = ndarray.array([[2, 2, 2], [2, 2, 2]])  # (2, 3)
c = a * b
d = c.transpose(1, 0)
d.materialize()
print(d.tolist())
