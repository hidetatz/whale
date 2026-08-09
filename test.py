import math
import unittest

import backend
import materialize
import ndarray
from dtype import float64, int64

class Test(unittest.TestCase):
    def setUp(self):
        materialize.reset()

    def _assert_list_close(self, got, expected, places=6):
        if isinstance(expected, (int, float)):
            self.assertAlmostEqual(got, expected, places=places)
        else:
            self.assertEqual(len(got), len(expected))
            for g, e in zip(got, expected):
                self._assert_list_close(g, e, places=places)
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

    def test_truediv(self):
        a = ndarray.array([6, 4, 2])
        b = ndarray.array([2, 2, 2])
        c = a / b
        c.materialize()
        self.assertEqual(c.tolist(), [3, 2, 1])

    def test_truediv_fractional(self):
        a = ndarray.array([1.0, 3.0, 5.0])
        b = ndarray.array([2.0, 2.0, 2.0])
        c = a / b
        c.materialize()
        self._assert_list_close(c.tolist(), [0.5, 1.5, 2.5])

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
    # reduce (keepdims)
    #

    def test_sum_keepdims_axis0(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.sum(axis=0, keepdims=True)
        b.materialize()
        self.assertEqual(b.shape, (1, 3))
        self.assertEqual(b.tolist(), [[5, 7, 9]])

    def test_sum_keepdims_axis1(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.sum(axis=1, keepdims=True)
        b.materialize()
        self.assertEqual(b.shape, (2, 1))
        self.assertEqual(b.tolist(), [[6], [15]])

    def test_sum_keepdims_multi_axis(self):
        a = ndarray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2,2,2)
        b = a.sum(axis=(0, 2), keepdims=True)
        b.materialize()
        self.assertEqual(b.shape, (1, 2, 1))
        self.assertEqual(b.tolist(), [[[1+2+5+6], [3+4+7+8]]])

    #
    # view
    #

    def test_view_materialize(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        a.materialize()
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(len(materialize.materializer.kern_invoke_hist), 0)

        a = a.reshape(3, 2)
        a.materialize()
        self.assertEqual(a.shape, (3, 2))
        self.assertEqual(a.tolist(), [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(materialize.materializer.kern_invoke_hist), 0)

        a = a.reshape(1, 6)
        a.materialize()
        self.assertEqual(a.shape, (1, 6))
        self.assertEqual(a.tolist(), [[1, 2, 3, 4, 5, 6]])
        self.assertEqual(len(materialize.materializer.kern_invoke_hist), 0)

        a = a.transpose(1, 0)
        a.materialize()
        self.assertEqual(a.shape, (6, 1))
        self.assertEqual(a.tolist(), [[1], [2], [3], [4], [5], [6]])
        self.assertEqual(len(materialize.materializer.kern_invoke_hist), 0)

        # contiguous needed
        a = a.reshape(2, 3)
        a.materialize()
        self.assertEqual(a.shape, (2, 3))
        self.assertEqual(a.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(len(materialize.materializer.kern_invoke_hist), 1)

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
    # transpose
    #

    def test_transpose_2d(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.transpose(1, 0)
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.strides, (1, 3))
        self.assertEqual(b.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_transpose_3d(self):
        a = ndarray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2,2,2)
        b = a.transpose(2, 0, 1)
        self.assertEqual(b.shape, (2, 2, 2))
        self.assertEqual(b.tolist(), [[[1, 3], [5, 7]], [[2, 4], [6, 8]]])

    def test_T_property_2d(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.T
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_T_property_3d(self):
        a = ndarray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2,2,2)
        b = a.T
        self.assertEqual(b.shape, (2, 2, 2))
        self.assertEqual(b.tolist(), [[[1, 5], [3, 7]], [[2, 6], [4, 8]]])

    def test_T_1d_is_self(self):
        a = ndarray.arange(5)
        b = a.T
        self.assertIs(b, a)

    def test_double_transpose(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.T.T
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_transpose_no_materialize(self):
        # pure view transpose of a constant does not need materialize
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.transpose(1, 0)
        self.assertEqual(b.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_transpose_then_add(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])    # (2,3)
        b = ndarray.array([[1, 1], [1, 1], [1, 1]])  # (3,2)
        c = a.T + b
        c.materialize()
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.tolist(), [[2, 5], [3, 6], [4, 7]])

    def test_transpose_then_sum_axis0(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = a.T  # (3,2): [[1,4],[2,5],[3,6]]
        c = b.sum(axis=0)  # sum rows: [1+2+3, 4+5+6] = [6, 15]
        c.materialize()
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c.tolist(), [6, 15])

    def test_transpose_then_sum_axis1(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = a.T  # (3,2): [[1,4],[2,5],[3,6]]
        c = b.sum(axis=1)  # sum cols: [1+4, 2+5, 3+6] = [5, 7, 9]
        c.materialize()
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.tolist(), [5, 7, 9])

    def test_transpose_wrong_axes_error(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(RuntimeError):
            a.transpose(0, 2)  # invalid: 2 is out of range for ndim=2

    #
    # reshape
    #

    def test_reshape_1d_to_2d(self):
        a = ndarray.arange(6)
        b = a.reshape(2, 3)
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_reshape_2d_to_1d(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.reshape(6)
        self.assertEqual(b.shape, (6,))
        self.assertEqual(b.tolist(), [1, 2, 3, 4, 5, 6])

    def test_reshape_2d_to_2d(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = a.reshape(3, 2)
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.tolist(), [[1, 2], [3, 4], [5, 6]])

    def test_reshape_no_materialize(self):
        # pure reshape of a contiguous constant does not need materialize
        a = ndarray.array([[0, 1, 2], [3, 4, 5]])
        b = a.reshape(6)
        self.assertEqual(b.tolist(), [0, 1, 2, 3, 4, 5])

    def test_reshape_then_sum(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.reshape(6).sum()
        b.materialize()
        self.assertEqual(b.tolist(), 21)

    def test_reshape_then_add(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([1, 1, 1, 1, 1, 1])
        c = a.reshape(6) + b
        c.materialize()
        self.assertEqual(c.tolist(), [2, 3, 4, 5, 6, 7])

    def test_noncontiguous_reshape(self):
        # transpose makes non-contiguous; reshape forces contiguous copy first
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = a.T   # (3,2): [[1,4],[2,5],[3,6]]
        c = b.reshape(6)  # flat in row-major of transposed: [1,4,2,5,3,6]
        c.materialize()
        self.assertEqual(c.shape, (6,))
        self.assertEqual(c.tolist(), [1, 4, 2, 5, 3, 6])

    def test_reshape_wrong_size_error(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(RuntimeError):
            a.reshape(5)

    def test_reshape_3d(self):
        a = ndarray.arange(24)
        b = a.reshape(2, 3, 4)
        self.assertEqual(b.shape, (2, 3, 4))
        self.assertEqual(b.tolist()[0][0], [0, 1, 2, 3])
        self.assertEqual(b.tolist()[1][2], [20, 21, 22, 23])

    #
    # contiguous
    #

    def test_is_contiguous_for_array(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(a.is_contiguous())

    def test_is_contiguous_for_transposed(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.T
        self.assertFalse(b.is_contiguous())

    def test_contiguous_already_contiguous_is_same(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.contiguous()
        self.assertIs(b, a)

    def test_contiguous_of_transposed(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.T  # (3,2): [[1,4],[2,5],[3,6]]
        c = b.contiguous()
        c.materialize()
        self.assertEqual(c.shape, (3, 2))
        self.assertTrue(c.is_contiguous())
        self.assertEqual(c.tolist(), [[1, 4], [2, 5], [3, 6]])

    #
    # float dtype
    #

    def test_float_array_dtype(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        self.assertEqual(a.dtype, float64)
        self.assertEqual(a.shape, (3,))
        self.assertEqual(a.tolist(), [1.0, 2.0, 3.0])

    def test_float_2d_array(self):
        a = ndarray.array([[1.5, 2.5], [3.5, 4.5]])
        self.assertEqual(a.dtype, float64)
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(a.tolist(), [[1.5, 2.5], [3.5, 4.5]])

    def test_float_add(self):
        a = ndarray.array([1.5, 2.5, 3.5])
        b = ndarray.array([0.5, 0.5, 0.5])
        c = a + b
        c.materialize()
        self.assertEqual(c.tolist(), [2.0, 3.0, 4.0])

    def test_float_sub(self):
        a = ndarray.array([3.0, 5.0, 7.0])
        b = ndarray.array([1.5, 2.5, 3.5])
        c = a - b
        c.materialize()
        self.assertEqual(c.tolist(), [1.5, 2.5, 3.5])

    def test_float_mul(self):
        a = ndarray.array([1.5, 2.0, 4.0])
        b = ndarray.array([2.0, 3.0, 0.5])
        c = a * b
        c.materialize()
        self.assertEqual(c.tolist(), [3.0, 6.0, 2.0])

    def test_float_neg(self):
        a = ndarray.array([1.5, -2.5, 3.0])
        b = -a
        b.materialize()
        self.assertEqual(b.tolist(), [-1.5, 2.5, -3.0])

    def test_float_pow(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        b = ndarray.array([2.0, 2.0, 2.0])
        c = a ** b
        c.materialize()
        self.assertEqual(c.tolist(), [1.0, 4.0, 9.0])

    def test_float_sum(self):
        a = ndarray.array([[1.5, 2.5, 3.0], [4.0, 5.0, 6.0]])
        b = a.sum(axis=1)
        b.materialize()
        self._assert_list_close(b.tolist(), [7.0, 15.0])

    def test_int_array_dtype(self):
        a = ndarray.array([1, 2, 3])
        self.assertEqual(a.dtype, int64)

    #
    # large arrays
    #

    def test_large_1d_add(self):
        n = 1000
        a = ndarray.arange(n)
        b = ndarray.arange(n)
        c = a + b
        c.materialize()
        expected = [i * 2 for i in range(n)]
        self.assertEqual(c.tolist(), expected)

    def test_large_1d_sum(self):
        n = 1000
        a = ndarray.arange(n)
        b = a.sum()
        b.materialize()
        self.assertEqual(b.tolist(), n * (n - 1) // 2)

    def test_large_2d_sum(self):
        # (100, 10) array of all 1s, sum along axis 1
        a = ndarray.full((100, 10), 1)
        b = a.sum(axis=1)
        b.materialize()
        self.assertEqual(b.shape, (100,))
        self.assertEqual(b.tolist(), [10] * 100)

    def test_large_float_mul(self):
        n = 500
        vals = [float(i) for i in range(n)]
        a = ndarray.array(vals)
        b = ndarray.full_like(a, 2.0)
        c = a * b
        c.materialize()
        self._assert_list_close(c.tolist(), [v * 2.0 for v in vals])

    #
    # chained operations
    #

    def test_chain_add_add(self):
        a = ndarray.arange(4)
        b = ndarray.arange(4)
        c = ndarray.arange(4)
        d = (a + b) + c
        d.materialize()
        self.assertEqual(d.tolist(), [0, 3, 6, 9])

    def test_chain_neg_add(self):
        a = ndarray.arange(4)
        b = ndarray.arange(4)
        c = (-a) + b
        c.materialize()
        self.assertEqual(c.tolist(), [0, 0, 0, 0])

    def test_mul_then_sum(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = ndarray.array([[2, 2], [2, 2]])
        c = (a * b).sum()
        c.materialize()
        self.assertEqual(c.tolist(), 20)

    def test_transpose_reshape_sum(self):
        # replicates example.py: mul -> transpose -> reshape -> sum
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = ndarray.array([[2, 2, 2], [2, 2, 2]])
        c = (a * b).transpose(1, 0).reshape(6).sum()
        c.materialize()
        self.assertEqual(c.tolist(), 42)

    def test_chain_with_broadcast(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        b = ndarray.array([1, 2, 3])                # (3,) -> broadcast to (2,3)
        c = (a + b).sum(axis=0)  # [[2,4,6],[5,7,9]] -> sum -> [7,11,15]
        c.materialize()
        self.assertEqual(c.tolist(), [7, 11, 15])

    def test_reuse_intermediate(self):
        # same intermediate used in two different ops
        a = ndarray.arange(4)
        b = ndarray.arange(4)
        mid = a + b
        c = mid * mid  # (a+b)^2
        c.materialize()
        self.assertEqual(c.tolist(), [0, 4, 16, 36])

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
