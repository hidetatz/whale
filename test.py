import unittest

import ndarray

class Test(unittest.TestCase):
    def test_construction(self):
        with self.subTest("array"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            self.assertEqual(a.shape, (2, 3))
            self.assertEqual(a.strides, (3, 1))
            self.assertEqual(a.offset, 0)
            self.assertEqual(a.ndim, 2)
            self.assertEqual(a.tolist(), [1, 2, 3, 4, 5, 6])

        with self.subTest("arange"):
            a = ndarray.arange(6)
            self.assertEqual(a.shape, (6,))
            self.assertEqual(a.strides, (1,))
            self.assertEqual(a.offset, 0)
            self.assertEqual(a.ndim, 1)
            self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5])

        with self.subTest("full"):
            a = ndarray.full((2, 3), 5)
            self.assertEqual(a.shape, (2, 3))
            self.assertEqual(a.strides, (3, 1))
            self.assertEqual(a.offset, 0)
            self.assertEqual(a.ndim, 2)
            self.assertEqual(a.tolist(), [5, 5, 5, 5, 5, 5])

        with self.subTest("full_like"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = ndarray.full_like(a, 5)
            self.assertEqual(b.shape, (2, 3))
            self.assertEqual(b.tolist(), [5, 5, 5, 5, 5, 5])

        with self.subTest("ones_like"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = ndarray.ones_like(a)
            self.assertEqual(b.shape, (2, 3))
            self.assertEqual(b.tolist(), [1, 1, 1, 1, 1, 1])

        with self.subTest("zeros_like"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = ndarray.zeros_like(a)
            self.assertEqual(b.shape, (2, 3))
            self.assertEqual(b.tolist(), [0, 0, 0, 0, 0, 0])

    def test_unary(self):
        with self.subTest("neg"):
            a = ndarray.arange(6)
            b = -a
            b.materialize()
            self.assertEqual(b.tolist(), [0, -1, -2, -3, -4, -5])

        with self.subTest("neg -> neg"):
            a = ndarray.arange(6)
            b = -a
            c = -b
            c.materialize()
            self.assertEqual(c.tolist(), [0, 1, 2, 3, 4, 5])

    def test_binary(self):
        with self.subTest("add"):
            a = ndarray.arange(6)
            b = ndarray.arange(6)
            c = a + b
            c.materialize()
            self.assertEqual(c.tolist(), [0, 2, 4, 6, 8, 10])

        with self.subTest("sub"):
            a = ndarray.arange(6)
            b = ndarray.arange(6)
            c = a - b
            c.materialize()
            self.assertEqual(c.tolist(), [0, 0, 0, 0, 0, 0])

        with self.subTest("mul"):
            a = ndarray.arange(6)
            b = ndarray.full_like(a, 2)
            c = a * b
            c.materialize()
            self.assertEqual(c.tolist(), [0, 2, 4, 6, 8, 10])

        with self.subTest("pow"):
            a = ndarray.arange(6)
            b = ndarray.full_like(a, 2)
            c = a ** b
            c.materialize()
            self.assertEqual(c.tolist(), [0, 1, 4, 9, 16, 25])

        with self.subTest("add -> sub -> mul -> div"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = ndarray.array([[1, 2, 3], [1, 2, 3]])
            c = ndarray.array([[2, 2, 2], [2, 2, 2]])
            d = ndarray.array([[1, 2, 3], [4, 5, 6]])
            e = ndarray.array([[2, 2, 2], [2, 2, 2]])
            f = a + b - c * d / e
            f.materialize()
            self.assertEqual(f.tolist(), [1, 2, 3, 1, 2, 3])

    def test_reduce(self):
        with self.subTest("sum 0"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = a.sum(axis=0)
            b.materialize()
            self.assertEqual(b.tolist(), [5, 7, 9])
            self.assertEqual(b.shape, (3,))

        with self.subTest("sum 1"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = a.sum(axis=1)
            b.materialize()
            self.assertEqual(b.tolist(), [6, 15])
            self.assertEqual(b.shape, (2,))

        with self.subTest("sum 0,1"):
            a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
            b = a.sum(axis=(0, 1))
            b.materialize()
            self.assertEqual(b.tolist(), [10, 14, 18])
            self.assertEqual(b.shape, (3,))

        with self.subTest("sum 0,2"):
            a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
            b = a.sum(axis=(0, 2))
            b.materialize()
            self.assertEqual(b.tolist(), [12, 30])
            self.assertEqual(b.shape, (2,))

        with self.subTest("sum 1,2"):
            a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
            b = a.sum(axis=(1, 2))
            b.materialize()
            self.assertEqual(b.tolist(), [21, 21])
            self.assertEqual(b.shape, (2,))

        with self.subTest("sum all"):
            a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
            b = a.sum()
            b.materialize()
            self.assertEqual(b.tolist(), [42])
            self.assertEqual(b.shape, ())

        with self.subTest("add -> sum"):
            a = ndarray.array([[1, 2, 3], [4, 5, 6]])
            b = ndarray.array([[1, 2, 3], [1, 2, 3]])
            # a + b -> [[2, 4, 6], [5, 7, 9]]
            c = (a + b).sum(axis=0)
            c.materialize()
            self.assertEqual(c.tolist(), [7, 11, 15])

if __name__ == '__main__':
    unittest.main()
