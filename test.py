import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
    def assert_almost_eq(self, l, r):
        if isinstance(l, int) or isinstance(l, float):
            assert isinstance(r, int) or isinstance(r, float), f"l and r type difference: {l=}, {r=}"
            if isinstance(r, int) or isinstance(r, float):
                self.assertAlmostEqual(l, r)
                return

        assert len(l) == len(r), f"l and r length difference: {l=}, {r=}"
        for i in range(len(l)):
            self.assert_almost_eq(l[i], r[i])

    def test_arith(self):
        results = {}
        for mod in [tensor, torch]:
            t1 = mod.tensor([1.0, 2.0, 3.0])
            t2 = mod.tensor([4.0, 5.0, 6.0])
            t3 = mod.tensor([7.0, 8.0, 9.0])
            t4 = mod.tensor([10.0, 11.0, 12.0])
            t5 = mod.tensor([13.0, 14.0, 15.0])
            t6 = mod.tensor([0.0, 2.0, 4.0])

            results[mod.__name__] = t1 + t2 * t3 - t4 / t5**t6

        self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist())

    def test_math(self):
        results = {}
        for mod in [tensor, torch]:
            t1 = mod.tensor(2.0)
            results[mod.__name__] = t1.log()

        self.assert_almost_eq(results["tensor"].tolist(), results["torch"].tolist())

    def test_crop(self):
        # crop and backprop
        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        t1 = t.crop((None, None, None))
        self.assert_almost_eq(t.tolist(), t1.tolist())
        t1.backward()
        self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        t1 = t.crop(((0, 1), None, None))
        self.assert_almost_eq(t1.tolist(), [[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]])
        t1.backward()
        self.assert_almost_eq(t.grad.tolist(), [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        t1 = t.crop(((0, 1), (1, 3), None))
        self.assert_almost_eq(t1.tolist(), [[[3, 4, 5], [6, 7, 8]]])
        t1.backward()
        self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        t1 = t.crop(((1, 2), (1, 3), (1, 3)))
        self.assert_almost_eq(t1.tolist(), [[[16, 17], [19, 20]]])
        t1.backward()
        self.assert_almost_eq(t.grad.tolist(), [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 1], [0, 1, 1], [0, 0, 0]]])

        # validation
        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        self.assertRaises(RuntimeError, t.crop, (None, None))

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        self.assertRaises(RuntimeError, t.crop, ((2, 3), None, None))

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        self.assertRaises(RuntimeError, t.crop, (None, (-1, 2), None))

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        self.assertRaises(RuntimeError, t.crop, (None, (1, 5), None))

        t = tensor.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]]])
        self.assertRaises(RuntimeError, t.crop, ((1, 0), None, None))

    def test_backprop(self):
        results = {}
        for mod in [tensor, torch]:
            t1 = mod.tensor(1.0, requires_grad=True)
            t2 = mod.tensor(4.0, requires_grad=True)
            t3 = mod.tensor(7.0, requires_grad=True)
            t4 = mod.tensor(10.0, requires_grad=True)
            t5 = mod.tensor(13.0, requires_grad=True)
            t6 = mod.tensor(3.0, requires_grad=True)
            t7 = mod.tensor(1.0, requires_grad=True)

            result = t1 + t2 * t3 - t4 / t5**t6 + t7
            result.backward()
            results[f"{mod.__name__}_grad_t1"] = t1.grad
            results[f"{mod.__name__}_grad_t2"] = t2.grad
            results[f"{mod.__name__}_grad_t3"] = t3.grad
            results[f"{mod.__name__}_grad_t4"] = t4.grad
            results[f"{mod.__name__}_grad_t5"] = t5.grad
            results[f"{mod.__name__}_grad_t6"] = t6.grad
            results[f"{mod.__name__}_grad_t7"] = t7.grad

        self.assert_almost_eq(results["tensor_grad_t1"].tolist(), results["torch_grad_t1"].tolist())
        self.assert_almost_eq(results["tensor_grad_t2"].tolist(), results["torch_grad_t2"].tolist())
        self.assert_almost_eq(results["tensor_grad_t3"].tolist(), results["torch_grad_t3"].tolist())
        self.assert_almost_eq(results["tensor_grad_t4"].tolist(), results["torch_grad_t4"].tolist())
        self.assert_almost_eq(results["tensor_grad_t5"].tolist(), results["torch_grad_t5"].tolist())
        self.assert_almost_eq(results["tensor_grad_t6"].tolist(), results["torch_grad_t6"].tolist())
        self.assert_almost_eq(results["tensor_grad_t7"].tolist(), results["torch_grad_t7"].tolist())


if __name__ == "__main__":
    unittest.main()
