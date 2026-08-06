import unittest

import materialize
import ndarray

class Test(unittest.TestCase):
    def setUp(self):
        materialize.reset()
    #
    # construction
    #

    def test_array(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.strides, (3, 1))
        self.assertEqual(a.offset, 0)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_arange(self):
        a = ndarray.arange(6)
        self.assertEqual(a.shape, (6,))
        self.assertEqual(a.strides, (1,))
        self.assertEqual(a.offset, 0)
        self.assertEqual(a.ndim, 1)
        self.assertEqual(a.tolist(), [0, 1, 2, 3, 4, 5])

    def test_full(self):
        a = ndarray.full((2, 3), 5)
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.strides, (3, 1))
        self.assertEqual(a.offset, 0)
        self.assertEqual(a.ndim, 2)
        self.assertEqual(a.tolist(), [[5, 5, 5], [5, 5, 5]])

    def test_full_like(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.full_like(a, 5)
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[5, 5, 5], [5, 5, 5]])

    def test_ones_like(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.ones_like(a)
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[1, 1, 1], [1, 1, 1]])

    def test_zeros_like(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.zeros_like(a)
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[0, 0, 0], [0, 0, 0]])

    #
    # unary
    #

    def test_neg(self):
        a = ndarray.arange(6)
        b = -a
        b.materialize()
        self.assertEqual(b.tolist(), [0, -1, -2, -3, -4, -5])

    def test_neg_to_neg(self):
        a = ndarray.arange(6)
        b = -a
        c = -b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 1, 2, 3, 4, 5])

    #
    # binary
    #

    def test_add(self):
        a = ndarray.arange(6)
        b = ndarray.arange(6)
        c = a + b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 2, 4, 6, 8, 10])

    def test_sub(self):
        a = ndarray.arange(6)
        b = ndarray.arange(6)
        c = a - b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 0, 0, 0, 0, 0])

    def test_mul(self):
        a = ndarray.arange(6)
        b = ndarray.full_like(a, 2)
        c = a * b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 2, 4, 6, 8, 10])

    def test_pow(self):
        a = ndarray.arange(6)
        b = ndarray.full_like(a, 2)
        c = a ** b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 1, 4, 9, 16, 25])

    def test_add_sub_mul_div(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([[1, 2, 3], [1, 2, 3]])
        c = ndarray.array([[2, 2, 2], [2, 2, 2]])
        d = ndarray.array([[1, 2, 3], [4, 5, 6]])
        e = ndarray.array([[2, 2, 2], [2, 2, 2]])
        f = a + b - c * d / e
        f.materialize()
        self.assertEqual(f.tolist(), [[1, 2, 3], [1, 2, 3]])

    def test_add_same_arr(self):
        a = ndarray.arange(6)
        b = a + a
        b.materialize()
        self.assertEqual(b.tolist(), [0, 2, 4, 6, 8, 10])

    #
    # reduce
    #

    def test_sum_0(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.sum(axis=0)
        b.materialize()
        self.assertEqual(b.tolist(), [5, 7, 9])
        self.assertEqual(b.shape, (3,))

    def test_sum_1(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.sum(axis=1)
        b.materialize()
        self.assertEqual(b.tolist(), [6, 15])
        self.assertEqual(b.shape, (2,))

    def test_sum_0_1(self):
        a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
        b = a.sum(axis=(0, 1))
        b.materialize()
        self.assertEqual(b.tolist(), [10, 14, 18])
        self.assertEqual(b.shape, (3,))

    def test_sum_0_2(self):
        a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
        b = a.sum(axis=(0, 2))
        b.materialize()
        self.assertEqual(b.tolist(), [12, 30])
        self.assertEqual(b.shape, (2,))

    def test_sum_1_2(self):
        a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
        b = a.sum(axis=(1, 2))
        b.materialize()
        self.assertEqual(b.tolist(), [21, 21])
        self.assertEqual(b.shape, (2,))

    def test_sum_all(self):
        a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
        b = a.sum()
        b.materialize()
        self.assertEqual(b.tolist(), 42)
        self.assertEqual(b.shape, ())

    def test_add_to_sum(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([[1, 2, 3], [1, 2, 3]])
        # a + b -> [[2, 4, 6], [5, 7, 9]]
        c = (a + b).sum(axis=0)
        c.materialize()
        self.assertEqual(c.tolist(), [7, 11, 15])

    #
    # broadcast
    #

    def test_2_3_mul_3_to_2_3(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([2, 2, 2])
        c = a * b
        c.materialize()
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.tolist(), [[2, 4, 6], [8, 10, 12]])

    def test_2_3_mul_1_1_to_2_3(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([[2]])
        c = a * b
        c.materialize()
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.tolist(), [[2, 4, 6], [8, 10, 12]])

    def test_2_3_mul_1_1_1_to_1_2_3(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([[[2]]])
        c = a * b
        c.materialize()
        self.assertEqual(c.shape, (1, 2, 3))
        self.assertEqual(c.tolist(), [[[2, 4, 6], [8, 10, 12]]])

    def test_sum_to_broadcast(self):
        a = ndarray.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])  # (2, 2, 3)
        b = a.sum(axis=(1,)) # [[5, 7, 9], [5, 7, 9]]
        c = b * ndarray.array([2])
        c.materialize()
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.tolist(), [[10, 14, 18], [10, 14, 18]])

    def test_2_3_mul_2_to_error(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([2, 2])
        with self.assertRaises(RuntimeError) as context:
            c = a * b
        self.assertTrue("shapes are not broadcastable" in str(context.exception))

    #
    # cache
    #

    def test_kern_cache(self):
        a = ndarray.arange(6)
        b = ndarray.arange(6)
        c = a + b
        c.materialize()

        d = ndarray.arange(6)
        e = ndarray.arange(6)
        f = d + e
        f.materialize()
        self.assertEqual(f.tolist(), [0, 2, 4, 6, 8, 10])
        self.assertEqual(list(materialize.materializer.kern_cache.hitcnt.values())[0], 1)

if __name__ == '__main__':
    unittest.main()
