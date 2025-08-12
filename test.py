import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
    def get_tensor(self, mod, arr):
        return tensor.array(arr) if mod == "whale" else torch.tensor(arr, requires_grad=True)

    def backprop(self, mod, t):
        return t.backprop() if mod == "whale" else t.backward()

    def calc(self, mod, t):
        return t.materialize() if mod == "whale" else t

    def test_arith(self):
        results = {}
        for mod in ["whale", "torch"]:
            t1 = self.get_tensor(mod, [1.0, 2.0, 3.0])
            t2 = self.get_tensor(mod, [4.0, 5.0, 6.0])
            t3 = self.get_tensor(mod, [7.0, 8.0, 9.0])
            t4 = self.get_tensor(mod, [10.0, 11.0, 12.0])
            t5 = self.get_tensor(mod, [13.0, 14.0, 15.0])
            t6 = self.get_tensor(mod, [0.0, 2.0, 4.0])

            results[mod] = self.calc(mod, t1 + t2 * t3 - t4 / t5**t6)

        self.assertEqual(results["whale"].tolist(), results["torch"].tolist())

    def test_backprop(self):
        results = {}
        for mod in ["whale", "torch"]:
            t1 = self.get_tensor(mod, 1.0)
            t2 = self.get_tensor(mod, 4.0)
            t3 = self.get_tensor(mod, 7.0)
            t4 = self.get_tensor(mod, 10.0)
            t5 = self.get_tensor(mod, 13.0)

            result = self.calc(mod, t1 + t2 * t3 - t4 / t5)
            self.backprop(mod, result)
            results[f"{mod}_grad_t1"] = self.calc(mod, t1.grad)
            results[f"{mod}_grad_t2"] = self.calc(mod, t2.grad)
            results[f"{mod}_grad_t3"] = self.calc(mod, t3.grad)
            results[f"{mod}_grad_t4"] = self.calc(mod, t4.grad)
            results[f"{mod}_grad_t5"] = self.calc(mod, t5.grad)

        self.assertEqual(results["whale_grad_t1"].tolist()[0], results["torch_grad_t1"].tolist())
        self.assertEqual(results["whale_grad_t2"].tolist()[0], results["torch_grad_t2"].tolist())
        self.assertEqual(results["whale_grad_t3"].tolist()[0], results["torch_grad_t3"].tolist())
        self.assertEqual(results["whale_grad_t4"].tolist()[0], results["torch_grad_t4"].tolist())
        self.assertEqual(results["whale_grad_t5"].tolist()[0], results["torch_grad_t5"].tolist())


if __name__ == "__main__":
    unittest.main()
