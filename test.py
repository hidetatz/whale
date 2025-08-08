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

        tt1 = torch.Tensor([1, 2, 3])
        tt2 = torch.Tensor([4, 5, 6])
        tt3 = torch.Tensor([7, 8, 9])

        tresult = tt1 + tt2 * tt3

        self.assertEqual(wresult.data, tresult.tolist())


if __name__ == "__main__":
    unittest.main()
