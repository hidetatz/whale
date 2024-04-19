import numpy as np

def toidx(shp, target):
    def product(args):
        r = 1
        for arg in args:
            r *= arg
        return r

    strides = []
    for i in range(len(shp)):
        strides.append(product(shp[i+1:]))

    idx = []
    sum = 0
    for i in range(len(shp)):
        for j in reversed(range(shp[i])):
            if sum + strides[i] * j > target:
                continue

            idx.append(j)
            sum += strides[i] * j
            break
    return idx

x = np.arange(0, 648).reshape(3, 6, 6, 6, 1)
y = x[0:2:1, 3, 1:6:1, [1, 5, 2], [0, 0, 0]]
# y = x[0:2:1, 3, 2, 1:6:1, [0, 0, 0]]

# x = np.arange(0, 24).reshape(4, 3, 2)
# y = x[0, 0:, [0, 1]]
# y = x[0:, 1:, [0, 1]]

data = y.flatten()
shp = y.shape
# print("x:", x)
print("shape:", shp)
print("data:", data)
for d in data:
    print(f"{d}: {toidx(x.shape, d)}")
