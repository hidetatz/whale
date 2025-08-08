import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
    def test_arith(self):
        wt1 = tensor.array([1, 2, 3])
        wt2 = tensor.array([4, 5, 6])
        wt3 = tensor.array([7, 8, 9])

        wresult = wt1 + wt2 * wt3
        wresult.materialize()

        tt1 = torch.tensor([1, 2, 3])
        tt2 = torch.tensor([4, 5, 6])
        tt3 = torch.tensor([7, 8, 9])

        tresult = tt1 + tt2 * tt3

        self.assertEqual(wresult.data, tresult.tolist())

    def test_backprop(self):
        wt1 = tensor.array(1)
        wt2 = tensor.array(4)
        wt3 = tensor.array(7)

        wresult = wt1 + wt2 * wt3
        wresult.backprop()
        wt1.grad.materialize()
        wt2.grad.materialize()
        wt3.grad.materialize()

        tt1 = torch.tensor(1.0, requires_grad=True)
        tt2 = torch.tensor(4.0, requires_grad=True)
        tt3 = torch.tensor(7.0, requires_grad=True)

        tresult = tt1 + tt2 * tt3
        tresult.backward()

        # fix
        self.assertEqual(wt1.grad.data[0], tt1.grad.tolist())
        self.assertEqual(wt2.grad.data[0], tt2.grad.tolist())
        self.assertEqual(wt3.grad.data[0], tt3.grad.tolist())


if __name__ == "__main__":
    unittest.main()
