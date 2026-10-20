import datetime
import time
import random
import struct

import ndarray
import materialize
import backend

orig_codegen = backend.bcknd.codegen
orig_compile = backend.bcknd.compile
orig_invoke = backend.bcknd.invoke_kernel

compile_count = [0]
hit_count = [0]
codegen_time = [0.0]
compile_time = [0.0]
invoke_time = [0.0]

def timed_codegen(name, func, schedule, extra):
    t = time.perf_counter()
    result = orig_codegen(name, func, schedule, extra)
    codegen_time[0] += time.perf_counter() - t
    return result

def timed_compile(name, code):
    t = time.perf_counter()
    result = orig_compile(name, code)
    compile_count[0] += 1
    compile_time[0] += time.perf_counter() - t
    return result

def timed_invoke(schedule, kern, params):
    t = time.perf_counter()
    result = orig_invoke(schedule, kern, params)
    invoke_time[0] += time.perf_counter() - t
    return result

backend.bcknd.codegen = timed_codegen
backend.bcknd.compile = timed_compile
backend.bcknd.invoke_kernel = timed_invoke

class MNIST:
    def __init__(self, root, train=True):
        prefix = "train" if train else "t10k"

        # images
        with open(f"{root}/{prefix}-images-idx3-ubyte", "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            images_bin = list(f.read()) # uint8

        images_floats = [x / 255.0 for x in images_bin]
        self.images = ndarray.array(images_floats).reshape(n, 28, 28)

        # labels
        with open(f"{root}/{prefix}-labels-idx1-ubyte", "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            labels_bin = list(f.read()) # uint8

        self.labels = ndarray.array(labels_bin)

    def __len__(self): return self.images.shape[0]
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

class Module:
    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, ndarray.ndarray): params.append(v)
            elif isinstance(v, Module): params.extend(v.parameters())
        return params

    def __call__(self, *args): return self.forward(*args)

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = ndarray.randn(in_features, out_features, requires_grad=True) * 0.01
        self.bias = ndarray.full((out_features,), 0.0, requires_grad=True)

    def forward(self, x): return x @ self.weight + self.bias
    def parameters(self): return [self.weight, self.bias]

class ReLU(Module):
    def forward(self, x): return x.relu()
    def parameters(self): return []

class Sequential(Module):
    def __init__(self, *mods):
        self.mods = mods

    def forward(self, x):
        for m in self.mods: x = m(x)
        return x

    def parameters(self):
        params = []
        for m in self.mods: params.extend(m.parameters())
        return params

class CrossEntropyLoss(Module):
    def forward(self, logits, labels):
        n = logits.shape[0]

        # softmax
        x = logits - logits.max(axis=1, keepdims=True)
        log_probs = x - x.exp().sum(axis=1, keepdims=True).log()

        # correct answer
        y = log_probs[ndarray.arange(n), labels]

        return -y.sum() / n

class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params: p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        super().__init__(params)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.materialize()  # detach grad from the backward graph
                p -= p.grad * self.lr
                p.materialize()

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        if shuffle: random.shuffle(self.indices)

    def __iter__(self):
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i+self.batch_size]
            # batch: [(image0, label0, xxx0...), (image1, label1, xxx1...), ...]
            batch = [self.dataset[j] for j in batch_indices]
            # zipped: [(image0, image1, ...), (label0, label1, ...), (xxx0, xxx1, ...), ...]
            zipped = list(zip(*batch))
            yield tuple(ndarray.stack(grp, dim=0) for grp in zipped)

def accuracy(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.reshape(images.shape[0], 784)
        logits = model(images)
        preds = logits.argmax(dim=1)
        preds.materialize()
        labels.materialize()
        correct += sum(p==l for p, l in zip(preds.tolist(), labels.tolist()))
        total += images.shape[0]
    return correct / total

def main():
    print("start mnist dataset load")

    train_dataset = MNIST("dataset/mnist", train=True)
    train_loader = DataLoader(train_dataset, batch_size=32)

    test_dataset = MNIST("dataset/mnist", train=False)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = Sequential(
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = CrossEntropyLoss()

    print("training start")

    for epoch in range(10):
        print(f"epoch: {epoch+1} {datetime.datetime.now()}")
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            print(f"minibatch: {i+1} {datetime.datetime.now()}")
            images = images.reshape(images.shape[0], 784)
            images.materialize()
            labels.materialize()
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            loss.materialize()
            total_loss += loss.tolist()

            mc = materialize.materializer.kern_cache
            hit = mc.total_hits - hit_count[0]
            print(f"compiles={compile_count[0]} hits={hit} codegen={codegen_time[0]:.2f}s compile={compile_time[0]:.2f}s invoke={invoke_time[0]:.2f}s")
            compile_count[0] = 0
            codegen_time[0] = compile_time[0] = invoke_time[0] = 0.0
            hit_count[0] = mc.total_hits

        # train_acc = accuracy(model, train_loader)
        # test_acc = accuracy(model, test_loader)
        # print(f"epoch {epoch+1}: loss={total_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
        print(f"epoch {epoch+1}: loss={total_loss:.4f} {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
