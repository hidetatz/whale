import math
import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
    #
    # helpers
    #
    def assert_almost_eq(self, l, r, places=7):
        if isinstance(l, int) or isinstance(l, float):
            assert isinstance(r, int) or isinstance(r, float), f"l and r type difference: {l=}, {r=}"
            if isinstance(r, int) or isinstance(r, float):
                if math.isnan(l):
                    self.assertTrue(math.isnan(r), f"left is nan but right is not, got {r}\n")
                else:
                    self.assertAlmostEqual(l, r, places=places)
                return

        assert len(l) == len(r), f"l and r length difference: {l=}, {r=}"
        for i in range(len(l)):
            self.assert_almost_eq(l[i], r[i], places=places)

    #
    # tests
    #
    def test_constructors(self):
        with self.subTest("arange"):
            t = tensor.Tensor.arange(6)
            self.assert_almost_eq(t.tolist(), [0, 1, 2, 3, 4, 5])

        with self.subTest("ones_like"):
            t2 = tensor.Tensor.ones_like(t)
            self.assert_almost_eq(t2.tolist(), [1, 1, 1, 1, 1, 1])

        with self.subTest("full_like"):
            t3 = tensor.Tensor.full_like(t2, 3)
            self.assert_almost_eq(t3.tolist(), [3, 3, 3, 3, 3, 3])

        with self.subTest("full"):
            t4 = tensor.Tensor.full((1, 2, 3), 5)
            self.assert_almost_eq(t4.tolist(), [[[5, 5, 5], [5, 5, 5]]])

    def test_operators(self):
        t = tensor.Tensor.arange(6)
        t1 = t + 3
        self.assert_almost_eq(t1.tolist(), [3, 4, 5, 6, 7, 8])

        t2 = 3 + t
        self.assert_almost_eq(t2.tolist(), [3, 4, 5, 6, 7, 8])

        t = tensor.Tensor.arange(6)
        t1 = t - 3
        self.assert_almost_eq(t1.tolist(), [-3, -2, -1, 0, 1, 2])
        t2 = 3 - t
        self.assert_almost_eq(t2.tolist(), [3, 2, 1, 0, -1, -2])

        t = tensor.Tensor.arange(6)
        t1 = t * 3
        self.assert_almost_eq(t1.tolist(), [0, 3, 6, 9, 12, 15])
        t2 = 3 * t
        self.assert_almost_eq(t2.tolist(), [0, 3, 6, 9, 12, 15])

        t = tensor.Tensor.arange(6)
        t1 = t / 3
        self.assert_almost_eq(t1.tolist(), [0 / 3, 1 / 3, 2 / 3, 3 / 3, 4 / 3, 5 / 3], places=6)
        t2 = 3 / t
        self.assert_almost_eq(t2.tolist(), [float("inf"), 3 / 1, 3 / 2, 3 / 3, 3 / 4, 3 / 5], places=6)

    def test_arith(self):
        with self.subTest("+, -, *, /, **"):
            results = {}
            for mod in [tensor, torch]:
                t1 = mod.tensor([1.0, 2.0, 3.0], requires_grad=True)
                t2 = mod.tensor([4.0, 5.0, 6.0], requires_grad=True)
                t3 = mod.tensor([7.0, 8.0, 9.0], requires_grad=True)
                t4 = mod.tensor([10.0, 11.0, 12.0], requires_grad=True)
                t5 = mod.tensor([13.0, 14.0, 15.0], requires_grad=True)
                t6 = mod.tensor([0.0, 2.0, 4.0], requires_grad=True)

                t7 = t1 + t2 * t3 - t4 / t5**t6
                results[mod.__name__] = t7
                t7.backward(gradient=mod.ones_like(t7))
                results[mod.__name__ + "_t1_grad"] = t1.grad
                results[mod.__name__ + "_t2_grad"] = t2.grad
                results[mod.__name__ + "_t3_grad"] = t3.grad
                results[mod.__name__ + "_t4_grad"] = t4.grad
                results[mod.__name__ + "_t5_grad"] = t5.grad
                results[mod.__name__ + "_t6_grad"] = t6.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist())
            self.assert_almost_eq(results["tensor_t1_grad"].tolist(), results["torch_t1_grad"].tolist())
            self.assert_almost_eq(results["tensor_t2_grad"].tolist(), results["torch_t2_grad"].tolist())
            self.assert_almost_eq(results["tensor_t3_grad"].tolist(), results["torch_t3_grad"].tolist())
            self.assert_almost_eq(results["tensor_t4_grad"].tolist(), results["torch_t4_grad"].tolist())
            self.assert_almost_eq(results["tensor_t5_grad"].tolist(), results["torch_t5_grad"].tolist())
            self.assert_almost_eq(results["tensor_t6_grad"].tolist(), results["torch_t6_grad"].tolist())

        with self.subTest("neg"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = -t
            self.assert_almost_eq(t1.tolist(), [[0, -1, -2], [-3, -4, -5]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[-1, -1, -1], [-1, -1, -1]])

    def test_compare(self):
        with self.subTest("ne, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 2], [0, 1, 2]])
            t3 = t1 != t2
            self.assert_almost_eq(t3.tolist(), [[False, False, False], [True, True, True]])

        with self.subTest("ne, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = t1 != 2
            self.assert_almost_eq(t2.tolist(), [[True, True, False], [True, True, True]])

        with self.subTest("eq, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 2], [0, 1, 2]])
            t3 = t1.eq(t2)
            self.assert_almost_eq(t3.tolist(), [[True, True, True], [False, False, False]])

        with self.subTest("eq, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = t1.eq(2)
            self.assert_almost_eq(t2.tolist(), [[False, False, True], [False, False, False]])

        with self.subTest("logical_not, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 2], [0, 1, 2]])
            t3 = t1 != t2
            t4 = t3.logical_not()
            self.assert_almost_eq(t4.tolist(), [[True, True, True], [False, False, False]])

        with self.subTest("<, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1 < t2
            self.assert_almost_eq(t3.tolist(), [[False, False, True], [True, False, False]])

        with self.subTest("<, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t3 = t1 < 2
            self.assert_almost_eq(t3.tolist(), [[True, True, False], [False, False, False]])

        with self.subTest(">, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1 > t2
            self.assert_almost_eq(t3.tolist(), [[False, False, False], [False, True, True]])

        with self.subTest(">, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t3 = t1 > 2
            self.assert_almost_eq(t3.tolist(), [[False, False, False], [True, True, True]])

        with self.subTest("<=, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1 <= t2
            self.assert_almost_eq(t3.tolist(), [[True, True, True], [True, False, False]])

        with self.subTest("<=, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t3 = t1 <= 2
            self.assert_almost_eq(t3.tolist(), [[True, True, True], [False, False, False]])

        with self.subTest(">=, simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1 >= t2
            self.assert_almost_eq(t3.tolist(), [[True, True, False], [False, True, True]])

        with self.subTest(">=, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t3 = t1 >= 2
            self.assert_almost_eq(t3.tolist(), [[False, False, True], [True, True, True]])

    def test_matmul(self):
        with self.subTest("2, 3 x 3 , 4"):
            results = {}
            for mod in [tensor, torch]:
                t1 = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t2 = mod.tensor([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]], requires_grad=True)

                t3 = t1 @ t2
                results[mod.__name__] = t3
                t3.backward(gradient=mod.ones_like(t3))
                results[mod.__name__ + "_t1_grad"] = t1.grad
                results[mod.__name__ + "_t2_grad"] = t2.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist())
            self.assert_almost_eq(results["tensor_t1_grad"].tolist(), results["torch_t1_grad"].tolist())
            self.assert_almost_eq(results["tensor_t2_grad"].tolist(), results["torch_t2_grad"].tolist())

        with self.subTest("not matrix"):
            t = tensor.tensor([[[1], [2]], [[3], [4]]])  # 2, 2, 1
            t2 = tensor.tensor([[1, 2], [3, 4]])  # 2, 3
            self.assertRaises(RuntimeError, t.matmul, t2)

        with self.subTest("invalid"):
            t1 = tensor.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3, 3
            t2 = tensor.tensor([[1, 2], [3, 4]])  # 2, 3
            self.assertRaises(RuntimeError, t.matmul, t2)

    def test_maximum(self):
        with self.subTest("simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1.maximum(t2)
            self.assert_almost_eq(t3.tolist(), [[0, 1, 3], [4, 4, 5]])
            t3.backprop()
            self.assert_almost_eq(t1.grad.tolist(), [[0, 0, 0], [0, 1, 1]])
            self.assert_almost_eq(t2.grad.tolist(), [[1, 1, 1], [1, 0, 0]])

        with self.subTest("simple, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = t1.maximum(2)
            self.assert_almost_eq(t2.tolist(), [[2, 2, 2], [3, 4, 5]])
            t2.backprop()
            self.assert_almost_eq(t1.grad.tolist(), [[0, 0, 0], [1, 1, 1]])

    def test_minimum(self):
        with self.subTest("simple"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = tensor.tensor([[0, 1, 3], [4, 3, 4]])
            t3 = t1.minimum(t2)
            self.assert_almost_eq(t3.tolist(), [[0, 1, 2], [3, 3, 4]])
            t3.backprop()
            self.assert_almost_eq(t1.grad.tolist(), [[0, 0, 1], [1, 0, 0]])
            self.assert_almost_eq(t2.grad.tolist(), [[1, 1, 0], [0, 1, 1]])

        with self.subTest("simple, broadcast"):
            t1 = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t2 = t1.minimum(2)
            self.assert_almost_eq(t2.tolist(), [[0, 1, 2], [2, 2, 2]])
            t2.backprop()
            self.assert_almost_eq(t1.grad.tolist(), [[1, 1, 0], [0, 0, 0]])

    def test_math(self):
        with self.subTest("log"):
            results = {}
            for mod in [tensor, torch]:
                t = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t1 = t.log()
                results[mod.__name__] = t1

                t1.backward(gradient=mod.ones_like(t1))
                results[mod.__name__ + "_grad"] = t.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist())
            self.assert_almost_eq(results["tensor_grad"].tolist(), results["torch_grad"].tolist())

        with self.subTest("sin"):
            results = {}
            for mod in [tensor, torch]:
                t = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t1 = t.sin()
                results[mod.__name__] = t1

                t1.backward(gradient=mod.ones_like(t1))
                results[mod.__name__ + "_grad"] = t.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist(), places=6)
            self.assert_almost_eq(results["tensor_grad"].tolist(), results["torch_grad"].tolist(), places=6)

        with self.subTest("cos"):
            results = {}
            for mod in [tensor, torch]:
                t = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t1 = t.cos()
                results[mod.__name__] = t1

                t1.backward(gradient=mod.ones_like(t1))
                results[mod.__name__ + "_grad"] = t.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist(), places=6)
            self.assert_almost_eq(results["tensor_grad"].tolist(), results["torch_grad"].tolist(), places=6)

        with self.subTest("tanh"):
            results = {}
            for mod in [tensor, torch]:
                t = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t1 = t.tanh()
                results[mod.__name__] = t1

                t1.backward(gradient=mod.ones_like(t1))
                results[mod.__name__ + "_grad"] = t.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist(), places=6)
            self.assert_almost_eq(results["tensor_grad"].tolist(), results["torch_grad"].tolist(), places=6)

        with self.subTest("exp"):
            results = {}
            for mod in [tensor, torch]:
                t = mod.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
                t1 = t.exp()
                results[mod.__name__] = t1

                t1.backward(gradient=mod.ones_like(t1))
                results[mod.__name__ + "_grad"] = t.grad

            self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist(), places=6)
            self.assert_almost_eq(results["tensor_grad"].tolist(), results["torch_grad"].tolist(), places=6)

    def test_sum(self):
        with self.subTest("2, 4, 3 -> sum(axis=0)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=0)
                self.assert_almost_eq(t1.tolist(), [[12, 14, 16], [18, 20, 22], [24, 26, 28], [30, 32, 34]])  # (4, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=0, keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[12, 14, 16], [18, 20, 22], [24, 26, 28], [30, 32, 34]]])  # (1, 4, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=1)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=1)
                self.assert_almost_eq(t1.tolist(), [[18, 22, 26], [66, 70, 74]])  # (2, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=1, keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[18, 22, 26]], [[66, 70, 74]]])  # (2, 1, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=2)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=2)
                self.assert_almost_eq(t1.tolist(), [[3, 12, 21, 30], [39, 48, 57, 66]])  # (2, 4)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=2, keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[3], [12], [21], [30]], [[39], [48], [57], [66]]])  # (2, 4, 1)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=(0, 1))"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(0, 1))
                self.assert_almost_eq(t1.tolist(), [84, 92, 100])  # (3,)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(1, 0), keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[84, 92, 100]]])  # (1, 1, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=(0, 2))"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(0, 2))
                self.assert_almost_eq(t1.tolist(), [42, 60, 78, 96])  # (4,)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(2, 0), keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[42], [60], [78], [96]]])  # (1, 4, 1)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=(1, 2))"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(1, 2))
                self.assert_almost_eq(t1.tolist(), [66, 210])  # (2,)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(2, 1), keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[66]], [[210]]])  # (2, 1, 1)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> sum(axis=(0, 1, 2))"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(0, 1, 2))
                self.assert_almost_eq(t1.tolist(), 276)  # ()
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(axis=(2, 1, 0), keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[276]]])  # (1, 1, 1)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("4 or more dims tensors"):
            with self.subTest("4 dims"):
                t = tensor.Tensor.arange(24).reshape(2, 2, 3, 2)
                t1 = t.sum(axis=(1, 3))
                self.assert_almost_eq(t1.tolist(), [[14.0, 22.0, 30.0], [62.0, 70.0, 78.0]])
                t1.backprop()
                self.assert_almost_eq(
                    t.grad.tolist(), [[[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]], [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]]]
                )

        with self.subTest("2, 4, 3 -> no axis"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum()
                self.assert_almost_eq(t1.tolist(), 276)  # ()
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.sum(keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[276]]])  # (1, 1, 1)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("sum on cropped"):
            with self.subTest("axis=0"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=0)
                self.assert_almost_eq(t2.tolist(), [[16, 17], [19, 20], [22, 23]])  # (3, 2)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=0, keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=0, keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[[16, 17], [19, 20], [22, 23]]])  # (1, 3, 2)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=1"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(1,))
                self.assert_almost_eq(t2.tolist(), [[57, 60]])  # (1, 2)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=1, keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(1,), keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[[57, 60]]])  # (1, 1, 2)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=2"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(2,))
                self.assert_almost_eq(t2.tolist(), [[33, 39, 45]])  # (1, 3)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=2, keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(2,), keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[[33], [39], [45]]])  # (1, 3, 1)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=2, 0"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(2, 0))
                self.assert_almost_eq(t2.tolist(), [33, 39, 45])  # (3,)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=2, 0, keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(axis=(2, 0), keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[[33], [39], [45]]])  # (1, 3, 1)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=None"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum()
                self.assert_almost_eq(t2.tolist(), 117)
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

            with self.subTest("axis=None, keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.crop(((1, 2), (1, 4), (1, 3)))
                t2 = t1.sum(keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[[117]]])
                t2.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])

        with self.subTest("sum on padded"):
            with self.subTest("axis=0"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                self.assert_almost_eq(
                    t1.tolist(), [[0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 2, 0], [0, 0, 3, 4, 5, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
                )  # (5, 6)
                t2 = t1.sum(axis=0)
                self.assert_almost_eq(t2.tolist(), [0, 0, 3, 5, 7, 0])
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=0, keepdims=True"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=0, keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[0, 0, 3, 5, 7, 0]])  # (1, 6)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=1"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=1)
                self.assert_almost_eq(t2.tolist(), [0, 3, 12, 0, 0])  # (5,)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=1, keepdims=True"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=1, keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[0], [3], [12], [0], [0]])  # (5, 1)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=(0, 1)"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=(0, 1))
                self.assert_almost_eq(t2.tolist(), 15)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=(0, 1), keepdims=True"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=(1, 0), keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[15]])  # (1, 1)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=None"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=None)
                self.assert_almost_eq(t2.tolist(), 15)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

            with self.subTest("axis=None, keepdims=True"):
                t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
                t1 = t.pad(((1, 2), (2, 1)))
                t2 = t1.sum(axis=None, keepdims=True)
                self.assert_almost_eq(t2.tolist(), [[15]])  # (1, 1)
                t2.backprop()
                self.assert_almost_eq(
                    t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
                )
                self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("4 or more dims sum"):
            t = tensor.tensor(
                [[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]], [[[12, 13], [14, 15]], [[16, 17], [18, 19]], [[20, 21], [22, 23]]]]
            )  # (2, 2, 3, 2)
            t1 = t.sum(axis=(2, 0))
            self.assert_almost_eq(t1.tolist(), [[28.0, 32.0], [44.0, 48.0], [60.0, 64.0]])

        with self.subTest("invalid axis"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])  # (2, 3)
            self.assertRaises(RuntimeError, t.sum, 3)

    def test_prod(self):
        with self.subTest("2, 4, 3 -> prod(axis=0)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0.1, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.prod(axis=0)
                self.assert_almost_eq(t1.tolist(), [[1.2, 13, 28], [45, 64, 85], [108, 133, 160], [189, 220, 253]])  # (4, 3)
                t1.backprop()
                self.assert_almost_eq(
                    t.grad.tolist(),
                    [[[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]], [[0.1, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]],
                    places=6,
                )

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0.1, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.prod(axis=0, keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[1.2, 13, 28], [45, 64, 85], [108, 133, 160], [189, 220, 253]]])  # (1, 4, 3)
                t1.backprop()
                self.assert_almost_eq(
                    t.grad.tolist(),
                    [[[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]], [[0.1, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]],
                    places=6,
                )

    def test_max(self):
        with self.subTest("2, 4, 3 -> max(axis=0)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.max(axis=0)
                self.assert_almost_eq(t1.tolist(), [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]])  # (4, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.max(axis=0, keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])  # (4, 3)
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("2, 4, 3 -> max(axis=0, 1)"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.max(axis=(0, 1))
                self.assert_almost_eq(t1.tolist(), [21, 22, 23])
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]]])

            with self.subTest("keepdims=True"):
                t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
                t1 = t.max(axis=(1, 0), keepdims=True)
                self.assert_almost_eq(t1.tolist(), [[[21, 22, 23]]])
                t1.backprop()
                self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]]])

        with self.subTest("same value"):
            with self.subTest("keepdims=False"):
                t = tensor.tensor([[[0, 1, 2], [3, 5, 5], [8, 8, 8], [9, 10, 11]], [[14, 13, 14], [15, 17, 17], [20, 20, 20], [21, 22, 23]]])
                t1 = t.max(axis=2)
                self.assert_almost_eq(t1.tolist(), [[2, 5, 8, 11], [14, 17, 20, 23]])
                t1.backprop()
                self.assert_almost_eq(
                    t.grad.tolist(),
                    [
                        [[0, 0, 1], [0, 0.5, 0.5], [0.33333333, 0.33333333, 0.33333333], [0, 0, 1]],
                        [[0.5, 0, 0.5], [0, 0.5, 0.5], [0.33333333, 0.33333333, 0.33333333], [0, 0, 1]],
                    ],
                )

    def test_transpose(self):
        with self.subTest("simple"):
            # (2, 3)
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.transpose()
            self.assert_almost_eq(t1.tolist(), [[0, 3], [1, 4], [2, 5]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("simple T"):
            # (2, 3)
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.T
            self.assert_almost_eq(t1.tolist(), [[0, 3], [1, 4], [2, 5]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("(2, 3, 2) -> (3, 2, 2)"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.transpose()
            self.assert_almost_eq(t1.tolist(), [[[0, 1], [6, 7]], [[2, 3], [8, 9]], [[4, 5], [10, 11]]])
            self.assertEqual(t1.shape, (3, 2, 2))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(2, 3, 2) -> (3, 2, 2) T"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.T
            self.assert_almost_eq(t1.tolist(), [[[0, 1], [6, 7]], [[2, 3], [8, 9]], [[4, 5], [10, 11]]])
            self.assertEqual(t1.shape, (3, 2, 2))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(2, 3, 2) -> (2, 3, 2)"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.transpose(2, 0)
            self.assert_almost_eq(t1.tolist(), [[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]])
            self.assertEqual(t1.shape, (2, 3, 2))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(2, 3, 2) -> (2, 3, 2) 2"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.transpose(0, 2)
            self.assert_almost_eq(t1.tolist(), [[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]])
            self.assertEqual(t1.shape, (2, 3, 2))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("invalid axes"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            self.assertRaises(RuntimeError, t.transpose, 4, 0)

        with self.subTest("invalid axes"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            self.assertRaises(RuntimeError, t.transpose, 2, -1)

    def test_permute(self):
        with self.subTest("no reorder"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.permute(0, 1, 2)
            self.assert_almost_eq(t1.tolist(), t.tolist())
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(0, 2, 1)"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.permute(0, 2, 1)
            self.assert_almost_eq(t1.tolist(), [[[0, 2, 4], [1, 3, 5]], [[6, 8, 10], [7, 9, 11]]])
            self.assertEqual(t1.shape, (2, 2, 3))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(1, 0, 2)"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.permute(1, 0, 2)
            self.assert_almost_eq(t1.tolist(), [[[0, 1], [6, 7]], [[2, 3], [8, 9]], [[4, 5], [10, 11]]])
            self.assertEqual(t1.shape, (3, 2, 2))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("(2, 0, 1)"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            t1 = t.permute(2, 0, 1)
            self.assert_almost_eq(t1.tolist(), [[[0, 2, 4], [6, 8, 10]], [[1, 3, 5], [7, 9, 11]]])
            self.assertEqual(t1.shape, (2, 2, 3))
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]])

        with self.subTest("permute cropped tensor"):
            # 2, 4, 3
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((1, 2), (1, 3), (0, 2)))  # (1, 2, 2), [15, 16, 18, 19]
            t2 = t1.permute(2, 0, 1)
            self.assert_almost_eq(t2.tolist(), [[[15, 18]], [[16, 19]]])
            self.assertEqual(t2.shape, (2, 1, 2))
            t2.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]]])

        with self.subTest("permute padded tensor"):
            t = tensor.tensor([[0, 1], [2, 3]])  # (2, 2)
            t1 = t.pad(((1, 2), (2, 1)))  # (5, 5)
            t2 = t1.permute(1, 0)
            self.assert_almost_eq(
                t2.tolist(),
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 1, 3, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
            )
            t2.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1], [1, 1]])

        with self.subTest("invalid axes"):
            # (2, 3, 2)
            t = tensor.tensor([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]])
            self.assertRaises(RuntimeError, t.permute, (2, 2, 0))

    def test_broadcast_to(self):
        with self.subTest("1, 1, 3 -> 1, 2, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((1, 2, 3))
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [0, 1, 2]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[2, 2, 2]]])

        with self.subTest("1, 1, 3 -> 2, 1, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((2, 1, 3))
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2]], [[0, 1, 2]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[2, 2, 2]]])

        with self.subTest("1, 1, 3 -> 2, 2, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((2, 2, 3))
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[4, 4, 4]]])

        with self.subTest("1, 1, 3 -> 2, 1, 1, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((2, 1, 1, 3))
            self.assert_almost_eq(t1.tolist(), [[[[0, 1, 2]]], [[[0, 1, 2]]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[2, 2, 2]]])

        with self.subTest("1, 1, 3 -> 2, 1, 2, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((2, 1, 2, 3))
            self.assert_almost_eq(t1.tolist(), [[[[0, 1, 2], [0, 1, 2]]], [[[0, 1, 2], [0, 1, 2]]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[4, 4, 4]]])

        with self.subTest("1, 1, 3 -> 2, 2, 2, 3"):
            t = tensor.tensor([[[0, 1, 2]]])
            t1 = t.broadcast_to((2, 2, 2, 3))
            self.assert_almost_eq(t1.tolist(), [[[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]], [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]])
            t1.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[8, 8, 8]]])

        with self.subTest("broadcast cropped tensor"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((1, 2), (1, 3), (0, 2)))  # (1, 2, 2), [15, 16, 18, 19]
            t2 = t1.broadcast_to((2, 2, 2))
            self.assert_almost_eq(t2.tolist(), [[[15, 16], [18, 19]], [[15, 16], [18, 19]]])
            t2.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [2, 2, 0], [2, 2, 0], [0, 0, 0]]])

        with self.subTest("broadcast cropped tensor 2"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((1, 2), (1, 3), (0, 2)))  # (1, 2, 2), [15, 16, 18, 19]
            t2 = t1.broadcast_to((2, 2, 2, 2))
            self.assert_almost_eq(t2.tolist(), [[[[15, 16], [18, 19]], [[15, 16], [18, 19]]], [[[15, 16], [18, 19]], [[15, 16], [18, 19]]]])
            t2.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [4, 4, 0], [4, 4, 0], [0, 0, 0]]])

        with self.subTest("broadcast padded tensor"):
            t = tensor.tensor([[0, 1], [2, 3]])  # (2, 2)
            t1 = t.pad(((1, 2), (2, 1)))  # (5, 5)
            t2 = t1.broadcast_to((2, 2, 5, 5))
            self.assert_almost_eq(
                t2.tolist(),
                [
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 2, 3, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 2, 3, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    ],
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 2, 3, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                        [
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 2, 3, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                        ],
                    ],
                ],
            )
            t2.backprop()
            self.assert_almost_eq(t.grad.tolist(), [[4, 4], [4, 4]])

    def test_crop(self):
        with self.subTest("Nones"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop((None, None, None))
            self.assert_almost_eq(t.tolist(), t1.tolist())
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        with self.subTest("simple 1"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((0, 1), None, None))
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("simple 2"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((0, 1), (1, 3), None))
            self.assert_almost_eq(t1.tolist(), [[[3, 4, 5], [6, 7, 8]]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("simple 3"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((1, 2), (1, 3), (1, 3)))
            self.assert_almost_eq(t1.tolist(), [[[16, 17], [19, 20]]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]]])

        with self.subTest("crop on cropped"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t.crop(((0, 1), None, None))
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
            t2 = t1.crop(((0, 1), (1, 3), (0, 2)))
            self.assert_almost_eq(t2.tolist(), [[[3, 4], [6, 7]]])
            t2.backward()
            self.assert_almost_eq(t1.grad.tolist(), [[[0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]]])
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        # validation
        with self.subTest("too less args"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            self.assertRaises(RuntimeError, t.crop, (None, None))

        with self.subTest("too big"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            self.assertRaises(RuntimeError, t.crop, ((2, 3), None, None))

        with self.subTest("negative is not allowd"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            self.assertRaises(RuntimeError, t.crop, (None, (-1, 2), None))

        with self.subTest("too big"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            self.assertRaises(RuntimeError, t.crop, (None, (1, 5), None))

        with self.subTest("r < l is not allowed"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            self.assertRaises(RuntimeError, t.crop, ((1, 0), None, None))

    def test_pad(self):
        with self.subTest("Nones"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.pad((None, None))
            self.assert_almost_eq(t.tolist(), t1.tolist())
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("simple"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.pad(((0, 1), (1, 2)))
            self.assert_almost_eq(t1.tolist(), [[0, 0, 1, 2, 0, 0], [0, 3, 4, 5, 0, 0], [0, 0, 0, 0, 0, 0]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("pad on padded"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.pad(((0, 1), (1, 2)))
            self.assert_almost_eq(t1.tolist(), [[0, 0, 1, 2, 0, 0], [0, 3, 4, 5, 0, 0], [0, 0, 0, 0, 0, 0]])
            t2 = t1.pad(((1, 2), (2, 1)))
            self.assert_almost_eq(
                t2.tolist(),
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 2, 0, 0, 0],
                    [0, 0, 0, 3, 4, 5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            )
            t2.backward()
            self.assert_almost_eq(t1.grad.tolist(), [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        # validation
        with self.subTest("too many pads"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.pad, ((0, 0), (0, 0), (0, 0)))

        with self.subTest("too less pads"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.pad, ((-1, 0),))

        with self.subTest("(start, stop) needed"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.pad, ((0, 0), (0,)))
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.pad, ((0, 0), (0, 0, 0)))

    def test_reshape(self):
        with self.subTest("2, 3 -> 1, 6"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.reshape(1, 6)
            self.assert_almost_eq(t1.tolist(), [[0, 1, 2, 3, 4, 5]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("2, 3 -> 3, 2"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.reshape(3, 2)
            self.assert_almost_eq(t1.tolist(), [[0, 1], [2, 3], [4, 5]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        with self.subTest("reshape on reshaped"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            t1 = t.reshape(3, 2)
            self.assert_almost_eq(t1.tolist(), [[0, 1], [2, 3], [4, 5]])
            t2 = t1.reshape(6, 1)
            self.assert_almost_eq(t2.tolist(), [[0], [1], [2], [3], [4], [5]])
            t2.backward()
            self.assert_almost_eq(t1.grad.tolist(), [[1, 1], [1, 1], [1, 1]])
            self.assert_almost_eq(t.grad.tolist(), [[1, 1, 1], [1, 1, 1]])

        # validation
        with self.subTest("length incompatible"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.reshape, (2, 4))

        with self.subTest("empty is not allowed"):
            t = tensor.tensor([[0, 1, 2], [3, 4, 5]])
            self.assertRaises(RuntimeError, t.reshape, ())

    def test_getitem(self):
        with self.subTest("basic: int indexing 1"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[0]
            self.assert_almost_eq(t1.tolist(), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: int indexing 2"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[1, 2]
            self.assert_almost_eq(t1.tolist(), [18, 19, 20])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]]])

        with self.subTest("basic: int indexing 3"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[0, 1, 2]
            self.assert_almost_eq(t1.tolist(), 5)
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: slice indexing 1"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[0:1]
            self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])  # dimension is not reduced
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: slice indexing 2"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[0:1, 1:3]
            self.assert_almost_eq(t1.tolist(), [[[3, 4, 5], [6, 7, 8]]])  # dimension is not reduced
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: slice indexing 3"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[0:1, 1:3, 0:2]
            self.assert_almost_eq(t1.tolist(), [[[3, 4], [6, 7]]])  # dimension is not reduced
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: slice default processing"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[:1, 1:, :]
            self.assert_almost_eq(t1.tolist(), [[[3, 4, 5], [6, 7, 8], [9, 10, 11]]])  # dimension is not reduced
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        with self.subTest("basic: int and slice mixed indexing"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[1, 1:3]
            self.assert_almost_eq(t1.tolist(), [[15, 16, 17], [18, 19, 20]])
            t1.backward()
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]]])

        with self.subTest("basic: index on index"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[1, :3]
            self.assert_almost_eq(t1.tolist(), [[12, 13, 14], [15, 16, 17], [18, 19, 20]])
            t2 = t1[1:, 2]
            self.assert_almost_eq(t2.tolist(), [17, 20])
            t2.backward()
            self.assert_almost_eq(t1.grad.tolist(), [[0, 0, 0], [0, 0, 1], [0, 0, 1]])
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]]])

        with self.subTest("basic: index * index"):
            t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
            t1 = t[1, :, 1:]
            self.assert_almost_eq(t1.tolist(), [[13, 14], [16, 17], [19, 20], [22, 23]])
            t2 = tensor.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
            t3 = t1 * t2
            self.assert_almost_eq(t3.tolist(), [[0, 14], [32, 51], [76, 100], [132, 161]])
            t3.backward()
            self.assert_almost_eq(t3.grad.tolist(), [[1, 1], [1, 1], [1, 1], [1, 1]])
            self.assert_almost_eq(t2.grad.tolist(), [[13, 14], [16, 17], [19, 20], [22, 23]])
            self.assert_almost_eq(t1.grad.tolist(), [[0, 1], [2, 3], [4, 5], [6, 7]])
            self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 2, 3], [0, 4, 5], [0, 6, 7]]])


if __name__ == "__main__":
    unittest.main()
