import ndarray

a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
b = a.sum()
b.materialize()
print(b.tolist())

