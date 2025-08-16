import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
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

        self.assertEqual(results["tensor"].tolist(), results["torch"].tolist())

    def test_view(self):
        results = {}
        for mod in [tensor, torch]:
            t1 = mod.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
            t2 = mod.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], [[11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22]]])
            t3 = t1[0, 1]
            t4 = t2[1, 2]
            results[mod.__name__] = t3 + t4

        self.assertEqual(results["tensor"].tolist(), results["torch"].tolist())

    def test_backprop(self):
        results = {}
        for mod in [tensor, torch]:
            t1 = mod.tensor(1.0, requires_grad=True)
            t2 = mod.tensor(4.0, requires_grad=True)
            t3 = mod.tensor(7.0, requires_grad=True)
            t4 = mod.tensor(10.0, requires_grad=True)
            t5 = mod.tensor(13.0, requires_grad=True)

            result = t1 + t2 * t3 - t4 / t5
            result.backward()
            results[f"{mod.__name__}_grad_t1"] = t1.grad
            results[f"{mod.__name__}_grad_t2"] = t2.grad
            results[f"{mod.__name__}_grad_t3"] = t3.grad
            results[f"{mod.__name__}_grad_t4"] = t4.grad
            results[f"{mod.__name__}_grad_t5"] = t5.grad

        self.assertEqual(results["tensor_grad_t1"].tolist(), results["torch_grad_t1"].tolist())
        self.assertEqual(results["tensor_grad_t2"].tolist(), results["torch_grad_t2"].tolist())
        self.assertEqual(results["tensor_grad_t3"].tolist(), results["torch_grad_t3"].tolist())
        self.assertEqual(results["tensor_grad_t4"].tolist(), results["torch_grad_t4"].tolist())
        self.assertEqual(results["tensor_grad_t5"].tolist(), results["torch_grad_t5"].tolist())


if __name__ == "__main__":
    unittest.main()
