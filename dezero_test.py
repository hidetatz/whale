import dezero
import dezero.functions as F
import numpy as np

def p(name, d):
    print(f"{name}: data: {tuple(d.flatten())}, shape: {d.shape}, strides: {d.strides}")

# reshape
x = dezero.Variable(np.arange(1, 13).reshape(2, 2, 3))
y = x.reshape(6, 2)
y.backward()
p("reshape y", y.data)
p("reshape x.grad", x.grad.data)
print("-------")

# transpose
x = dezero.Variable(np.arange(1, 13).reshape(2, 2, 3))
y = x.transpose()
y.backward()
p("transpose y", y.data)
p("transpose x.grad", x.grad.data)
print("-------")

# broadcast_to
x = dezero.Variable(np.arange(1, 7).reshape(2, 3))
y = F.broadcast_to(x, (3, 2, 3))
y.backward()
p("broadcast_to y", y.data)
p("broadcast_to x.grad", x.grad.data)
print("-------")

# sum
x = dezero.Variable(np.arange(1, 7).reshape(2, 3))
y = F.sum(x)
y.backward()
p("sum y", y.data)
p("sum x.grad", x.grad.data)
print("-------")

# sum_to
x = dezero.Variable(np.arange(1, 7).reshape(2, 3))
y = F.sum_to(x, (1, 3))
y.backward()
p("sum_to y", y.data)
p("sum_to x.grad", x.grad.data)
print("-------")

# exp
x = dezero.Variable(np.arange(1, 7).reshape(2, 3))
y = F.exp(x)
y.backward()
p("exp y", y.data)
p("exp x.grad", x.grad.data)
print("-------")

# add
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.arange(7, 13).reshape(2, 3))
y = x1 + x2
y.backward()
p("add y", y.data)
p("add x1.grad", x1.grad.data)
p("add x2.grad", x2.grad.data)
print("-------")

# add2
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.array(2))
y = x1 + x2
y.backward()
p("add2 y", y.data)
p("add2 x1.grad", x1.grad.data)
p("add2 x2.grad", x2.grad.data)
print("-------")

# sub
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.arange(7, 13).reshape(2, 3))
y = x1 - x2
y.backward()
p("sub y", y.data)
p("sub x1.grad", x1.grad.data)
p("sub x2.grad", x2.grad.data)
print("-------")

# sub2
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.array(2))
y = x1 - x2
y.backward()
p("sub2 y", y.data)
p("sub2 x1.grad", x1.grad.data)
p("sub2 x2.grad", x2.grad.data)
print("-------")

# mul
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.arange(7, 13).reshape(2, 3))
y = x1 * x2
y.backward()
p("mul y", y.data)
p("mul x1.grad", x1.grad.data)
p("mul x2.grad", x2.grad.data)
print("-------")

# mul2
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.array(2))
y = x1 * x2
y.backward()
p("mul2 y", y.data)
p("mul2 x1.grad", x1.grad.data)
p("mul2 x2.grad", x2.grad.data)
print("-------")

# div
x1 = dezero.Variable(np.array([[2, 4, 6], [8, 10, 12]]))
x2 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
y = x1 / x2
y.backward()
p("div y", y.data)
p("div x1.grad", x1.grad.data)
p("div x2.grad", x2.grad.data)
print("-------")

# div2
x1 = dezero.Variable(np.arange(1, 7).reshape(2, 3))
x2 = dezero.Variable(np.array(2))
y = x1 / x2
y.backward()
p("div2 y", y.data)
p("div2 x1.grad", x1.grad.data)
p("div2 x2.grad", x2.grad.data)
print("-------")
