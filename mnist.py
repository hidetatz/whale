import dataset
from dtype import dtypes
from tensor import Tensor

train_imgs, train_lbls, test_imgs, test_lbls = dataset.mnist()

# W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
w1 = Tensor.rand(784, 128) * Tensor(1 / 784).sqrt()
w2 = Tensor.rand(128, 10) * Tensor(1 / 128).sqrt()

batch_size = 100

train_data_offset = 0
def get_train_data():
    global train_data_offset
    global batch_size
    x_train = train_imgs[train_data_offset : train_data_offset + batch_size]
    y_train = train_lbls[train_data_offset : train_data_offset + batch_size]
    train_data_offset += batch_size
    return (
        Tensor(x_train, requires_grad=True),
        Tensor(y_train, requires_grad=True),
    )

test_data_offset = 0
def get_test_data():
    global test_data_offset
    global batch_size
    x_test = test_imgs[test_data_offset : test_data_offset + batch_size]
    y_test = test_lbls[test_data_offset : test_data_offset + batch_size]
    data_offset += batch_size
    return (
        Tensor(x_test, requires_grad=True),
        Tensor(y_test, requires_grad=True),
    )


def model(x, w1, w2):
    return ((x @ w1).relu()) @ w2


for i in range(5):
    b = 0
    while True:
        x_train, y_train = get_train_data()
        if not x_train:
            break

        y = model(x_train, w1, w2)

        loss = y.softmax_cross_entropy(y_train)

        w1.grad = None
        w2.grad = None
        loss.backward()

        assert w1.grad is not None
        assert w2.grad is not None

        # sgd
        nw1 = w1 - 0.01 * w1.grad
        nw2 = w2 - 0.01 * w2.grad

        print(f"minibatch {b} train_loss", loss.tolist())

        w1 = Tensor(nw1.tolist())
        w2 = Tensor(nw2.tolist())

        loss.free_all()
        b += 1

    while True:
        x_test, y_test = get_test_data()
        if not x_test:
            break

        acc = (model(x_test, w1, w2).argmax(axis=1).reshape(*y_test.shape).eq(y_test)).to(dtypes.float32).mean()
        print(f"testacc: {acc.item()}")
