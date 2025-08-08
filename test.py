import unittest

import torch

import tensor


class WhaleTest(unittest.TestCase):
    def test_arith(self):
        wt1 = tensor.array([1, 2, 3])
        wt2 = tensor.array([4, 5, 6])
        wt3 = tensor.array([7, 8, 9])
        wt4 = tensor.array([10, 11, 12])
        wt5 = tensor.array([13, 14, 15])

        wresult = wt1 + wt2 * wt3 - wt4 / wt5
        wresult.materialize()

        tt1 = torch.tensor([1, 2, 3])
        tt2 = torch.tensor([4, 5, 6])
        tt3 = torch.tensor([7, 8, 9])
        tt4 = torch.tensor([10, 11, 12])
        tt5 = torch.tensor([13, 14, 15])

        tresult = tt1 + tt2 * tt3 - tt4 / tt5

        self.assertEqual(wresult.data, tresult.tolist())

    def test_backprop(self):
        wt1 = tensor.array(1)
        wt2 = tensor.array(4)
        wt3 = tensor.array(7)
        wt4 = tensor.array(10)
        wt5 = tensor.array(13)

        wresult = wt1 + wt2 * wt3 - wt4 / wt5
        wresult.backprop()
        wt1.grad.materialize()
        wt2.grad.materialize()
        wt3.grad.materialize()
        wt4.grad.materialize()
        wt5.grad.materialize()

        tt1 = torch.tensor(1.0, requires_grad=True)
        tt2 = torch.tensor(4.0, requires_grad=True)
        tt3 = torch.tensor(7.0, requires_grad=True)
        tt4 = torch.tensor(10.0, requires_grad=True)
        tt5 = torch.tensor(13.0, requires_grad=True)

        tresult = tt1 + tt2 * tt3 - tt4 / tt5
        tresult.backward()

        # fix
        self.assertEqual(wt1.grad.data[0], tt1.grad.tolist())
        self.assertEqual(wt2.grad.data[0], tt2.grad.tolist())
        self.assertEqual(wt3.grad.data[0], tt3.grad.tolist())
        self.assertEqual(wt4.grad.data[0], tt4.grad.tolist())
        self.assertEqual(wt5.grad.data[0], tt5.grad.tolist())


if __name__ == "__main__":
    unittest.main()
