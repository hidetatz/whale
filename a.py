import numpy as np

x = np.array([0.43, 0.091]).reshape(1, 2, 1, 1)
print(x)
y = x[[[0, 0, 0], [0, 0, 0]], 0::1, :, [0, 0, 0]]
print(y.shape, len(y.flatten()))
print(y)
