import ndarray

a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
b = a.sum(axis=(1,)) # [[5, 7, 9], [5, 7, 9]]
c = b * ndarray.array([2])
c.materialize()
print(c.tolist())
