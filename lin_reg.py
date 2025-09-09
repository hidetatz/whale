from tensor import Tensor

def dataset():
    x = Tensor.rand(100, 1)
    y = 5 + 2 * x + Tensor.rand(100, 1)
    return x, y

W = Tensor.zeros((1, 1))
b = Tensor.zeros((1,))

def model(x):
    return (x @ W) + b

def mse(x0, x1):
    diff = x0 - x1
    return (diff ** 2).sum() / len(diff)

lr = 0.1
iters = 1000

x, y = dataset()

for i in range(iters):
    y_pred = model(x)
    loss = mse(y, y_pred)
    W.grad = None
    b.grad = None
    loss.backward()
    nW = W - lr * W.grad
    nb = b - lr * b.grad

    W = Tensor(nW.tolist())
    b = Tensor(nb.tolist())

    print(W.tolist(), b.tolist(), loss.tolist())
