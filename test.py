import math
import unittest

import materialize
import ndarray
from datatype import float64, int64

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
    # slice (basic indexing)
    #

    def test_slice_1d_int(self):
        a = ndarray.arange(6)
        self.assertEqual(a[2].shape, ())
        self.assertEqual(a[2].tolist(), 2)

    def test_slice_1d_negative_int(self):
        a = ndarray.arange(6)
        self.assertEqual(a[-1].tolist(), 5)
        self.assertEqual(a[-3].tolist(), 3)

    def test_slice_1d_basic(self):
        a = ndarray.arange(6)
        b = a[1:4]
        self.assertEqual(b.shape, (3,))
        self.assertEqual(b.strides, (1,))
        self.assertEqual(b.offset, 1)
        self.assertEqual(b.tolist(), [1, 2, 3])

    def test_slice_1d_from_start(self):
        a = ndarray.arange(6)
        self.assertEqual(a[:3].tolist(), [0, 1, 2])

    def test_slice_1d_to_end(self):
        a = ndarray.arange(6)
        self.assertEqual(a[3:].tolist(), [3, 4, 5])

    def test_slice_1d_step(self):
        a = ndarray.arange(6)
        b = a[::2]
        self.assertEqual(b.shape, (3,))
        self.assertEqual(b.strides, (2,))
        self.assertEqual(b.tolist(), [0, 2, 4])

    def test_slice_1d_start_step(self):
        a = ndarray.arange(6)
        self.assertEqual(a[1::2].tolist(), [1, 3, 5])

    def test_slice_1d_start_stop_step(self):
        a = ndarray.arange(6)
        self.assertEqual(a[1:5:2].tolist(), [1, 3])

    def test_slice_2d_int_int(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a[0, 0].tolist(), 1)
        self.assertEqual(a[1, 2].tolist(), 6)

    def test_slice_2d_first_int(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a[1]
        self.assertEqual(b.shape, (3,))
        self.assertEqual(b.tolist(), [4, 5, 6])

    def test_slice_2d_negative_int(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a[-1].tolist(), [4, 5, 6])
        self.assertEqual(a[-1, -1].tolist(), 6)

    def test_slice_2d_int_slice(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(a[0, 1:3].tolist(), [2, 3])
        self.assertEqual(a[1, :2].tolist(), [4, 5])

    def test_slice_2d_slice_int(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[1:3, 2]
        self.assertEqual(b.shape, (2,))
        self.assertEqual(b.tolist(), [6, 9])

    def test_slice_2d_slice_slice(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[1:3, 0:2]
        self.assertEqual(b.shape, (2, 2))
        self.assertEqual(b.tolist(), [[4, 5], [7, 8]])

    def test_slice_2d_step(self):
        a = ndarray.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        b = a[::2, ::2]
        self.assertEqual(b.shape, (2, 2))
        self.assertEqual(b.tolist(), [[1, 3], [9, 11]])

    def test_slice_no_materialize(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a[0:1, 1:]
        self.assertEqual(b.shape, (1, 2))
        self.assertEqual(b.tolist(), [[2, 3]])

    def test_slice_3d(self):
        a = ndarray.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2,2,2)
        self.assertEqual(a[0, :, 1].tolist(), [2, 4])
        self.assertEqual(a[1, 0].tolist(), [5, 6])

    def test_slice_then_add(self):
        a = ndarray.array([1, 2, 3, 4, 5, 6])
        b = ndarray.array([10, 10, 10])
        c = a[1:4] + b
        c.materialize()
        self.assertEqual(c.tolist(), [12, 13, 14])

    def test_slice_step_then_add(self):
        a = ndarray.arange(6)
        b = ndarray.array([1, 1, 1])
        c = a[1::2] + b
        c.materialize()
        self.assertEqual(c.tolist(), [2, 4, 6])

    def test_slice_then_sum(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[0:2].sum(axis=0)
        b.materialize()
        self.assertEqual(b.tolist(), [5, 7, 9])

    def test_slice_2d_then_mul(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = ndarray.array([[2, 2], [2, 2]])
        c = a[1:3, 0:2] * b
        c.materialize()
        self.assertEqual(c.tolist(), [[8, 10], [14, 16]])

    def test_slice_too_many_indices(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(RuntimeError):
            a[0, 1, 2]

    def test_slice_invalid_index_type(self):
        a = ndarray.arange(6)
        with self.assertRaises(RuntimeError):
            a[1.0]

    #
    # pad
    #

    def test_pad_1d(self):
        a = ndarray.array([1, 2, 3])
        b = a.pad(0, 2, 3)
        b.materialize()
        self.assertEqual(b.shape, (8,))
        self.assertEqual(b.tolist(), [0, 0, 1, 2, 3, 0, 0, 0])

    def test_pad_only_before(self):
        a = ndarray.array([1, 2, 3])
        b = a.pad(0, 2, 0)
        b.materialize()
        self.assertEqual(b.tolist(), [0, 0, 1, 2, 3])

    def test_pad_only_after(self):
        a = ndarray.array([1, 2, 3])
        b = a.pad(0, 0, 2)
        b.materialize()
        self.assertEqual(b.tolist(), [1, 2, 3, 0, 0])

    def test_pad_2d_dim0(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.pad(0, 1, 1)
        b.materialize()
        self.assertEqual(b.shape, (4, 2))
        self.assertEqual(b.tolist(), [[0, 0], [1, 2], [3, 4], [0, 0]])

    def test_pad_2d_dim1(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.pad(1, 1, 2)
        b.materialize()
        self.assertEqual(b.shape, (2, 5))
        self.assertEqual(b.tolist(), [[0, 1, 2, 0, 0], [0, 3, 4, 0, 0]])

    def test_pad_noop_returns_self(self):
        a = ndarray.arange(3)
        b = a.pad(0, 0, 0)
        self.assertIs(b, a)

    def test_backward_pad_1d(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        a.pad(0, 2, 3).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_pad_2d(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        a.pad(0, 1, 1).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_backward_pad_partial_gradient(self):
        # gradient only flows through the original (non-padded) positions
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        weights = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        (a.pad(0, 1, 2) * weights).sum().backward()
        a.grad.materialize()
        # a[0] at pos 1 (weight 2), a[1] at pos 2 (weight 3), a[2] at pos 3 (weight 4)
        self._assert_list_close(a.grad.tolist(), [2.0, 3.0, 4.0])

    #
    # dilate
    #

    def test_dilate_1d_step2(self):
        a = ndarray.array([1, 2, 3])
        b = a.dilate(0, 2)
        b.materialize()
        self.assertEqual(b.shape, (5,))
        self.assertEqual(b.tolist(), [1, 0, 2, 0, 3])

    def test_dilate_1d_step3(self):
        a = ndarray.array([1, 2, 3])
        b = a.dilate(0, 3)
        b.materialize()
        self.assertEqual(b.shape, (7,))
        self.assertEqual(b.tolist(), [1, 0, 0, 2, 0, 0, 3])

    def test_dilate_2d_dim0(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.dilate(0, 2)
        b.materialize()
        self.assertEqual(b.shape, (3, 2))
        self.assertEqual(b.tolist(), [[1, 2], [0, 0], [3, 4]])

    def test_dilate_2d_dim1(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.dilate(1, 2)
        b.materialize()
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[1, 0, 2], [3, 0, 4]])

    def test_dilate_noop_returns_self(self):
        a = ndarray.arange(3)
        b = a.dilate(0, 1)
        self.assertIs(b, a)

    def test_backward_dilate_step2(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        a.dilate(0, 2).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_dilate_step3(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        a.dilate(0, 3).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_dilate_2d(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        a.dilate(0, 2).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_backward_dilate_partial_gradient(self):
        # gradient flows only through original positions (not zeros inserted by dilate)
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        weights = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        (a.dilate(0, 2) * weights).sum().backward()
        a.grad.materialize()
        # a[0] at pos 0 (weight 1), a[1] at pos 2 (weight 3), a[2] at pos 4 (weight 5)
        self._assert_list_close(a.grad.tolist(), [1.0, 3.0, 5.0])

    #
    # backward
    #

    def test_backward_neg(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        (-a).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [-1.0, -1.0, -1.0])

    def test_backward_sin(self):
        a = ndarray.array([0.0, math.pi / 6, math.pi / 2], requires_grad=True)
        a.sin().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [math.cos(0.0), math.cos(math.pi / 6), math.cos(math.pi / 2)])

    def test_backward_cos(self):
        a = ndarray.array([0.0, math.pi / 6, math.pi / 2], requires_grad=True)
        a.cos().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [-math.sin(0.0), -math.sin(math.pi / 6), -math.sin(math.pi / 2)])

    def test_backward_exp(self):
        a = ndarray.array([0.0, 1.0, 2.0], requires_grad=True)
        a.exp().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [math.exp(0.0), math.exp(1.0), math.exp(2.0)])

    def test_backward_log(self):
        a = ndarray.array([1.0, 2.0, math.e], requires_grad=True)
        a.log().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 0.5, 1.0 / math.e])

    def test_backward_sqrt(self):
        a = ndarray.array([1.0, 4.0, 9.0], requires_grad=True)
        a.sqrt().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.5, 0.25, 1.0 / 6.0])

    def test_backward_add(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.array([4.0, 5.0, 6.0], requires_grad=True)
        (a + b).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])
        self._assert_list_close(b.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_sub(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.array([4.0, 5.0, 6.0], requires_grad=True)
        (a - b).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])
        self._assert_list_close(b.grad.tolist(), [-1.0, -1.0, -1.0])

    def test_backward_mul(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.array([2.0, 3.0, 4.0], requires_grad=True)
        (a * b).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [2.0, 3.0, 4.0])
        self._assert_list_close(b.grad.tolist(), [1.0, 2.0, 3.0])

    def test_backward_x_squared(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        (a * a).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [2.0, 4.0, 6.0])

    def test_backward_pow(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.full_like(a, 2.0)
        (a ** b).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [2.0, 4.0, 6.0])

    def test_backward_sum_all(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        a.sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_backward_sum_axis0(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        a.sum(axis=0).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_backward_sum_axis1(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        a.sum(axis=1).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_backward_reshape(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        a.reshape(2, 3).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_backward_transpose(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        a.T.sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_backward_broadcast(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], requires_grad=True)
        (a + b).sum().backward()
        a.grad.materialize()
        # a is broadcast across 2 rows, so each element accumulates gradient twice
        self._assert_list_close(a.grad.tolist(), [2.0, 2.0, 2.0])

    def test_backward_contiguous(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        a.T.contiguous().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_backward_slice_1d_basic(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        a[1:4].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_backward_slice_1d_from_end(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        a[3:].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_backward_slice_1d_step(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        a[::2].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    def test_backward_slice_1d_start_step(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        a[1:5:2].sum().backward()
        a.grad.materialize()
        # selects indices 1 and 3
        self._assert_list_close(a.grad.tolist(), [0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

    def test_backward_slice_int_index(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
        a[1].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    def test_backward_slice_2d_slice_slice(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
        a[1:3, 0:2].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    def test_backward_slice_2d_int_slice(self):
        a = ndarray.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)
        a[0, 1:4].sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])

    def test_backward_slice_weighted(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        weights = ndarray.array([3.0, 1.0, 4.0], requires_grad=True)
        (a[1:4] * weights).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 3.0, 1.0, 4.0, 0.0])

    def test_backward_chain_slice_mul(self):
        a = ndarray.array([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        b = ndarray.array([2.0, 3.0], requires_grad=True)
        (a[2:4] * b).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 0.0, 2.0, 3.0, 0.0])
        self._assert_list_close(b.grad.tolist(), [3.0, 4.0])

    #
    # cmp
    #

    def test_eq_same(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([1, 2, 3])
        c = a.equal(b)
        c.materialize()
        self.assertEqual(c.tolist(), [1, 1, 1])

    def test_eq_different(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([1, 0, 3])
        c = a.equal(b)
        c.materialize()
        self.assertEqual(c.tolist(), [1, 0, 1])

    def test_ne(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([1, 0, 3])
        c = a.not_equal(b)
        c.materialize()
        self.assertEqual(c.tolist(), [0, 1, 0])

    def test_gt(self):
        a = ndarray.array([3, 1, 2])
        b = ndarray.array([1, 2, 2])
        c = a.greater(b)
        c.materialize()
        self.assertEqual(c.tolist(), [1, 0, 0])

    def test_ge(self):
        a = ndarray.array([3, 1, 2])
        b = ndarray.array([1, 2, 2])
        c = a.greater_equal(b)
        c.materialize()
        self.assertEqual(c.tolist(), [1, 0, 1])

    def test_lt(self):
        a = ndarray.array([3, 1, 2])
        b = ndarray.array([1, 2, 2])
        c = a.less(b)
        c.materialize()
        self.assertEqual(c.tolist(), [0, 1, 0])

    def test_le(self):
        a = ndarray.array([3, 1, 2])
        b = ndarray.array([1, 2, 2])
        c = a.less_equal(b)
        c.materialize()
        self.assertEqual(c.tolist(), [0, 1, 1])

    def test_cmp_result_dtype_is_int64(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        b = ndarray.array([1.0, 0.0, 3.0])
        c = a.equal(b)
        self.assertEqual(c.dtype, int64)

    def test_cmp_with_broadcast(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([2])
        c = a.greater(b)
        c.materialize()
        self.assertEqual(c.tolist(), [0, 0, 1])

    def test_cmp_used_as_mask(self):
        a = ndarray.array([1.0, 2.0, 3.0, 2.0])
        b = ndarray.array([2.0])
        mask = a.equal(b).to(float64)
        result = a * mask
        result.materialize()
        self._assert_list_close(result.tolist(), [0.0, 2.0, 0.0, 2.0])

    def test_cmp_no_grad(self):
        # cmp ops always produce requires_grad=False; backward is a no-op
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        c = a.equal(b)
        self.assertFalse(c.requires_grad)
        c.materialize()
        c.backward()  # should not raise
        self.assertIsNone(a.grad)

    #
    # matmul
    #

    def test_matmul_square(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]])
        b = ndarray.array([[5.0, 6.0], [7.0, 8.0]])
        c = a @ b
        c.materialize()
        self._assert_list_close(c.tolist(), [[19.0, 22.0], [43.0, 50.0]])

    def test_matmul_rectangular(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = ndarray.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
        c = a @ b
        c.materialize()
        self.assertEqual(c.shape, (2, 2))
        self._assert_list_close(c.tolist(), [[4.0, 5.0], [10.0, 11.0]])

    def test_matmul_identity(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]])
        eye = ndarray.array([[1.0, 0.0], [0.0, 1.0]])
        c = a @ eye
        c.materialize()
        self._assert_list_close(c.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_matmul_shape_error(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]])
        b = ndarray.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with self.assertRaises(RuntimeError):
            a @ b

    def test_matmul_chained(self):
        a = ndarray.array([[1.0, 0.0], [0.0, 2.0]])
        b = ndarray.array([[3.0, 0.0], [0.0, 4.0]])
        c = ndarray.array([[1.0, 1.0], [1.0, 1.0]])
        d = (a @ b) @ c
        d.materialize()
        self._assert_list_close(d.tolist(), [[3.0, 3.0], [8.0, 8.0]])

    def test_backward_matmul_grad_a(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = ndarray.array([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        (a @ b).sum().backward()
        a.grad.materialize()
        # grad_A = grad_C @ B.T, grad_C = ones(2,2), B.T = [[5,7],[6,8]]
        self._assert_list_close(a.grad.tolist(), [[11.0, 15.0], [11.0, 15.0]])

    def test_backward_matmul_grad_b(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = ndarray.array([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        (a @ b).sum().backward()
        b.grad.materialize()
        # grad_B = A.T @ grad_C, A.T = [[1,3],[2,4]], grad_C = ones(2,2)
        self._assert_list_close(b.grad.tolist(), [[4.0, 4.0], [6.0, 6.0]])

    def test_backward_matmul_rectangular(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        b = ndarray.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
        (a @ b).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self.assertEqual(a.grad.shape, (2, 3))
        self.assertEqual(b.grad.shape, (3, 2))

    #
    # where
    #

    def test_where_basic(self):
        cond = ndarray.array([1, 0, 1, 0])
        x = ndarray.array([1.0, 2.0, 3.0, 4.0])
        y = ndarray.array([10.0, 20.0, 30.0, 40.0])
        z = ndarray.where(cond, x, y)
        z.materialize()
        self._assert_list_close(z.tolist(), [1.0, 20.0, 3.0, 40.0])

    def test_where_all_true(self):
        cond = ndarray.array([1, 1, 1])
        x = ndarray.array([1.0, 2.0, 3.0])
        y = ndarray.array([9.0, 9.0, 9.0])
        z = ndarray.where(cond, x, y)
        z.materialize()
        self._assert_list_close(z.tolist(), [1.0, 2.0, 3.0])

    def test_where_all_false(self):
        cond = ndarray.array([0, 0, 0])
        x = ndarray.array([1.0, 2.0, 3.0])
        y = ndarray.array([9.0, 9.0, 9.0])
        z = ndarray.where(cond, x, y)
        z.materialize()
        self._assert_list_close(z.tolist(), [9.0, 9.0, 9.0])

    def test_where_with_cmp(self):
        a = ndarray.array([1.0, 5.0, 3.0, 2.0])
        z = ndarray.where(a.greater(ndarray.array([3.0])), a, ndarray.full_like(a, 0.0))
        z.materialize()
        self._assert_list_close(z.tolist(), [0.0, 5.0, 0.0, 0.0])

    def test_where_broadcast_cond(self):
        cond = ndarray.array([1, 0])
        x = ndarray.array([[1.0, 2.0], [3.0, 4.0]])
        y = ndarray.array([[10.0, 20.0], [30.0, 40.0]])
        z = ndarray.where(cond, x, y)
        z.materialize()
        self._assert_list_close(z.tolist(), [[1.0, 20.0], [3.0, 40.0]])

    def test_where_result_dtype(self):
        cond = ndarray.array([1, 0])
        x = ndarray.array([1.0, 2.0])
        y = ndarray.array([3.0, 4.0])
        z = ndarray.where(cond, x, y)
        self.assertEqual(z.dtype, float64)

    def test_backward_where_x_grad(self):
        x = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        y = ndarray.array([10.0, 20.0, 30.0], requires_grad=True)
        cond = ndarray.array([1, 0, 1])
        z = ndarray.where(cond, x, y)
        z.sum().backward()
        x.grad.materialize()
        self._assert_list_close(x.grad.tolist(), [1.0, 0.0, 1.0])

    def test_backward_where_y_grad(self):
        x = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        y = ndarray.array([10.0, 20.0, 30.0], requires_grad=True)
        cond = ndarray.array([1, 0, 1])
        z = ndarray.where(cond, x, y)
        z.sum().backward()
        y.grad.materialize()
        self._assert_list_close(y.grad.tolist(), [0.0, 1.0, 0.0])

    def test_backward_where_weighted(self):
        x = ndarray.array([2.0, 3.0, 4.0], requires_grad=True)
        y = ndarray.array([10.0, 20.0, 30.0], requires_grad=True)
        cond = ndarray.array([1, 0, 1])
        w = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        (ndarray.where(cond, x, y) * w).sum().backward()
        x.grad.materialize()
        y.grad.materialize()
        self._assert_list_close(x.grad.tolist(), [1.0, 0.0, 3.0])
        self._assert_list_close(y.grad.tolist(), [0.0, 2.0, 0.0])

    #
    # max
    #

    def test_max_1d(self):
        a = ndarray.array([3.0, 1.0, 4.0, 1.0, 5.0])
        b = a.max()
        b.materialize()
        self._assert_list_close(b.tolist(), 5.0)

    def test_max_all_negative(self):
        a = ndarray.array([-3.0, -1.0, -4.0])
        b = a.max()
        b.materialize()
        self._assert_list_close(b.tolist(), -1.0)

    def test_max_axis0(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]])
        b = a.max(axis=0)
        b.materialize()
        self._assert_list_close(b.tolist(), [3.0, 5.0])

    def test_max_axis1(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]])
        b = a.max(axis=1)
        b.materialize()
        self._assert_list_close(b.tolist(), [5.0, 3.0])

    def test_max_keepdims(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]])
        b = a.max(axis=1, keepdims=True)
        b.materialize()
        self.assertEqual(b.shape, (2, 1))
        self._assert_list_close(b.tolist(), [[5.0], [3.0]])

    def test_backward_max_basic(self):
        a = ndarray.array([1.0, 5.0, 3.0], requires_grad=True)
        a.max().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 1.0, 0.0])

    def test_backward_max_ties(self):
        a = ndarray.array([3.0, 3.0, 1.0], requires_grad=True)
        a.max().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.5, 0.5, 0.0])

    def test_backward_max_axis0(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        a.max(axis=0).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 1.0], [1.0, 0.0]])

    def test_backward_max_axis1(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        a.max(axis=1).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 1.0], [1.0, 0.0]])

    def test_backward_max_keepdims(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]], requires_grad=True)
        a.max(axis=1, keepdims=True).sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[0.0, 1.0], [1.0, 0.0]])

    #
    # cast
    #

    def test_cast_int64_to_float64(self):
        a = ndarray.array([1, 2, 3])
        b = a.to(float64)
        b.materialize()
        self.assertEqual(b.dtype, float64)
        self._assert_list_close(b.tolist(), [1.0, 2.0, 3.0])

    def test_cast_float64_to_int64(self):
        a = ndarray.array([1.9, 2.5, 3.1])
        b = a.to(int64)
        b.materialize()
        self.assertEqual(b.dtype, int64)
        self.assertEqual(b.tolist(), [1, 2, 3])

    def test_cast_preserves_shape(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.to(float64)
        b.materialize()
        self.assertEqual(b.shape, (2, 2))
        self._assert_list_close(b.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_cast_noop_same_dtype(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        b = a.to(float64)
        b.materialize()
        self._assert_list_close(b.tolist(), [1.0, 2.0, 3.0])

    def test_cast_fused_with_subsequent_op(self):
        a = ndarray.array([1, 2, 3])
        b = a.to(float64) + ndarray.array([0.5, 0.5, 0.5])
        b.materialize()
        self._assert_list_close(b.tolist(), [1.5, 2.5, 3.5])

    def test_backward_cast_float64_to_float64(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        b = a.to(float64)
        b.sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_cast_int64_to_float64(self):
        a = ndarray.array([1, 2, 3], requires_grad=True)
        b = a.to(float64)
        b.sum().backward()
        a.grad.materialize()
        self.assertEqual(a.grad.dtype, int64)
        self.assertEqual(a.grad.tolist(), [1, 1, 1])

    #
    # inplace
    #

    def test_iadd_basic(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        a += ndarray.array([10.0, 20.0, 30.0])
        a.materialize()
        self._assert_list_close(a.tolist(), [11.0, 22.0, 33.0])

    def test_iadd_scalar(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        a += 5.0
        a.materialize()
        self._assert_list_close(a.tolist(), [6.0, 7.0, 8.0])

    def test_iadd_preserves_identity(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        original_id = id(a)
        a += ndarray.array([1.0, 1.0, 1.0])
        self.assertEqual(id(a), original_id)

    def test_iadd_twice(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        a += ndarray.array([1.0, 1.0, 1.0])
        a += ndarray.array([1.0, 1.0, 1.0])
        a.materialize()
        self._assert_list_close(a.tolist(), [3.0, 4.0, 5.0])

    def test_add_method(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        b = ndarray.array([4.0, 5.0, 6.0])
        c = a.add(b)
        c.materialize()
        self._assert_list_close(c.tolist(), [5.0, 7.0, 9.0])

    def test_add_inplace_method(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        original_id = id(a)
        a.add_(ndarray.array([10.0, 20.0, 30.0]))
        self.assertEqual(id(a), original_id)
        a.materialize()
        self._assert_list_close(a.tolist(), [11.0, 22.0, 33.0])

    def test_isub_basic(self):
        a = ndarray.array([10.0, 20.0, 30.0])
        a -= ndarray.array([1.0, 2.0, 3.0])
        a.materialize()
        self._assert_list_close(a.tolist(), [9.0, 18.0, 27.0])

    def test_isub_scalar(self):
        a = ndarray.array([10.0, 20.0, 30.0])
        a -= 5.0
        a.materialize()
        self._assert_list_close(a.tolist(), [5.0, 15.0, 25.0])

    def test_isub_preserves_identity(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        original_id = id(a)
        a -= ndarray.array([1.0, 1.0, 1.0])
        self.assertEqual(id(a), original_id)

    def test_isub_twice(self):
        a = ndarray.array([10.0, 20.0, 30.0])
        a -= ndarray.array([1.0, 1.0, 1.0])
        a -= ndarray.array([1.0, 1.0, 1.0])
        a.materialize()
        self._assert_list_close(a.tolist(), [8.0, 18.0, 28.0])

    def test_sub_inplace_method(self):
        a = ndarray.array([10.0, 20.0, 30.0])
        original_id = id(a)
        a.sub_(ndarray.array([1.0, 2.0, 3.0]))
        self.assertEqual(id(a), original_id)
        a.materialize()
        self._assert_list_close(a.tolist(), [9.0, 18.0, 27.0])

    def test_isub_sgd_like(self):
        # simulates SGD: w -= lr * grad
        w = ndarray.array([1.0, 2.0, 3.0])
        grad = ndarray.array([0.1, 0.2, 0.3])
        lr = 0.1
        w -= grad * lr
        w.materialize()
        self._assert_list_close(w.tolist(), [0.99, 1.98, 2.97])

    #
    # argmax
    #

    def test_argmax_1d(self):
        a = ndarray.array([1.0, 5.0, 3.0, 2.0])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), 1)

    def test_argmax_1d_first_elem(self):
        a = ndarray.array([9.0, 1.0, 2.0])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), 0)

    def test_argmax_1d_last_elem(self):
        a = ndarray.array([1.0, 2.0, 9.0])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), 2)

    def test_argmax_2d_dim0(self):
        a = ndarray.array([[1.0, 5.0], [3.0, 2.0]])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), [1, 0])

    def test_argmax_2d_dim1(self):
        a = ndarray.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        r = a.argmax(dim=1)
        r.materialize()
        self.assertEqual(r.tolist(), [1, 2])

    def test_argmax_negative_values(self):
        a = ndarray.array([-3.0, -1.0, -2.0])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), 1)

    def test_argmax_result_dtype(self):
        a = ndarray.array([1.0, 3.0, 2.0])
        r = a.argmax(dim=0)
        self.assertEqual(r.dtype, ndarray.int64)

    def test_argmax_batch(self):
        # typical MNIST usage: argmax of logits over class dim
        logits = ndarray.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        preds = logits.argmax(dim=1)
        preds.materialize()
        self.assertEqual(preds.tolist(), [1, 0, 2])

    def test_argmax_ties_returns_last(self):
        # tie-breaking: returns last occurrence (not first like PyTorch)
        a = ndarray.array([5.0, 3.0, 5.0])
        r = a.argmax(dim=0)
        r.materialize()
        self.assertEqual(r.tolist(), 2)

    #
    # relu
    #

    def test_relu_positive(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        r = a.relu()
        r.materialize()
        self._assert_list_close(r.tolist(), [1.0, 2.0, 3.0])

    def test_relu_negative(self):
        a = ndarray.array([-1.0, -2.0, -3.0])
        r = a.relu()
        r.materialize()
        self._assert_list_close(r.tolist(), [0.0, 0.0, 0.0])

    def test_relu_mixed(self):
        a = ndarray.array([-1.0, 0.0, 1.0, -2.0, 3.0])
        r = a.relu()
        r.materialize()
        self._assert_list_close(r.tolist(), [0.0, 0.0, 1.0, 0.0, 3.0])

    def test_relu_zero_grad(self):
        # gradient at x=0 should be 0, not 0.5
        a = ndarray.array([0.0], requires_grad=True)
        a.relu().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0])

    def test_relu_backward(self):
        a = ndarray.array([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        a.relu().sum().backward()
        a.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [0.0, 1.0, 0.0, 1.0])

    #
    # softmax
    #

    def test_softmax_sums_to_one(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        r = a.softmax(axis=0)
        r.materialize()
        self.assertAlmostEqual(sum(r.tolist()), 1.0, places=6)

    def test_softmax_values(self):
        a = ndarray.array([1.0, 2.0, 3.0])
        r = a.softmax(axis=0)
        r.materialize()
        vals = r.tolist()
        self.assertTrue(vals[0] < vals[1] < vals[2])

    def test_softmax_2d_axis1(self):
        a = ndarray.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        r = a.softmax(axis=1)
        r.materialize()
        rows = r.tolist()
        self.assertAlmostEqual(sum(rows[0]), 1.0, places=6)
        self.assertAlmostEqual(sum(rows[1]), 1.0, places=6)
        # uniform input -> uniform output
        self._assert_list_close(rows[1], [1/3, 1/3, 1/3])

    def test_softmax_numerical_stability(self):
        # large values should not produce inf
        a = ndarray.array([1000.0, 1001.0, 999.0])
        r = a.softmax(axis=0)
        r.materialize()
        vals = r.tolist()
        self.assertAlmostEqual(sum(vals), 1.0, places=6)
        for v in vals:
            self.assertFalse(math.isnan(v) or math.isinf(v))

    def test_softmax_backward(self):
        a = ndarray.array([1.0, 2.0, 3.0], requires_grad=True)
        r = a.softmax(axis=0)
        r.sum().backward()
        a.grad.materialize()
        # sum of softmax is always 1, so gradient w.r.t. each input is 0
        self._assert_list_close(a.grad.tolist(), [0.0, 0.0, 0.0])

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

    #
    # unsqueeze
    #

    def test_unsqueeze_dim0(self):
        a = ndarray.array([1, 2, 3])
        b = a.unsqueeze(0)
        self.assertEqual(b.shape, (1, 3))
        self.assertEqual(b.tolist(), [[1, 2, 3]])

    def test_unsqueeze_dim1(self):
        a = ndarray.array([1, 2, 3])
        b = a.unsqueeze(1)
        self.assertEqual(b.shape, (3, 1))
        self.assertEqual(b.tolist(), [[1], [2], [3]])

    def test_unsqueeze_2d_dim0(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.unsqueeze(0)
        self.assertEqual(b.shape, (1, 2, 2))
        self.assertEqual(b.tolist(), [[[1, 2], [3, 4]]])

    def test_unsqueeze_2d_dim1(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = a.unsqueeze(1)
        self.assertEqual(b.shape, (2, 1, 2))
        self.assertEqual(b.tolist(), [[[1, 2]], [[3, 4]]])

    def test_unsqueeze_is_reshape(self):
        # unsqueeze は reshape と同じ; materialize 不要
        a = ndarray.array([[1, 2, 3], [4, 5, 6]])
        b = a.unsqueeze(0)
        self.assertEqual(b.shape, (1, 2, 3))
        self.assertEqual(b.tolist(), [[[1, 2, 3], [4, 5, 6]]])

    #
    # cat / concat
    #

    def test_cat_1d(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([4, 5])
        c = ndarray.cat([a, b], dim=0)
        c.materialize()
        self.assertEqual(c.shape, (5,))
        self.assertEqual(c.tolist(), [1, 2, 3, 4, 5])

    def test_cat_2d_dim0(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = ndarray.array([[5, 6]])
        c = ndarray.cat([a, b], dim=0)
        c.materialize()
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.tolist(), [[1, 2], [3, 4], [5, 6]])

    def test_cat_2d_dim1(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = ndarray.array([[5], [6]])
        c = ndarray.cat([a, b], dim=1)
        c.materialize()
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.tolist(), [[1, 2, 5], [3, 4, 6]])

    def test_cat_three_arrays(self):
        a = ndarray.array([1, 2])
        b = ndarray.array([3, 4])
        c = ndarray.array([5, 6])
        d = ndarray.cat([a, b, c], dim=0)
        d.materialize()
        self.assertEqual(d.shape, (6,))
        self.assertEqual(d.tolist(), [1, 2, 3, 4, 5, 6])

    def test_cat_unequal_sizes(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([4])
        c = ndarray.cat([a, b], dim=0)
        c.materialize()
        self.assertEqual(c.shape, (4,))
        self.assertEqual(c.tolist(), [1, 2, 3, 4])

    def test_cat_ndim_mismatch_error(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([[4, 5, 6]])
        with self.assertRaises(ValueError):
            ndarray.cat([a, b], dim=0)

    def test_cat_shape_mismatch_error(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = ndarray.array([[5, 6, 7]])
        with self.assertRaises(ValueError):
            ndarray.cat([a, b], dim=0)

    def test_backward_cat_1d(self):
        a = ndarray.array([1.0, 2.0], requires_grad=True)
        b = ndarray.array([3.0, 4.0, 5.0], requires_grad=True)
        ndarray.cat([a, b], dim=0).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 1.0])
        self._assert_list_close(b.grad.tolist(), [1.0, 1.0, 1.0])

    def test_backward_cat_2d_dim0(self):
        a = ndarray.array([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = ndarray.array([[5.0, 6.0]], requires_grad=True)
        ndarray.cat([a, b], dim=0).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self._assert_list_close(b.grad.tolist(), [[1.0, 1.0]])

    def test_backward_cat_weighted(self):
        a = ndarray.array([1.0, 2.0], requires_grad=True)
        b = ndarray.array([3.0, 4.0], requires_grad=True)
        w = ndarray.array([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        (ndarray.cat([a, b], dim=0) * w).sum().backward()
        a.grad.materialize()
        b.grad.materialize()
        self._assert_list_close(a.grad.tolist(), [1.0, 2.0])
        self._assert_list_close(b.grad.tolist(), [3.0, 4.0])

    #
    # stack
    #

    def test_stack_1d_dim0(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([4, 5, 6])
        c = ndarray.stack([a, b], dim=0)
        c.materialize()
        self.assertEqual(c.shape, (2, 3))
        self.assertEqual(c.tolist(), [[1, 2, 3], [4, 5, 6]])

    def test_stack_1d_dim1(self):
        a = ndarray.array([1, 2, 3])
        b = ndarray.array([4, 5, 6])
        c = ndarray.stack([a, b], dim=1)
        c.materialize()
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.tolist(), [[1, 4], [2, 5], [3, 6]])

    def test_stack_three_1d(self):
        a = ndarray.array([1, 2])
        b = ndarray.array([3, 4])
        c = ndarray.array([5, 6])
        d = ndarray.stack([a, b, c], dim=0)
        d.materialize()
        self.assertEqual(d.shape, (3, 2))
        self.assertEqual(d.tolist(), [[1, 2], [3, 4], [5, 6]])

    def test_stack_2d_dim0(self):
        a = ndarray.array([[1, 2], [3, 4]])
        b = ndarray.array([[5, 6], [7, 8]])
        c = ndarray.stack([a, b], dim=0)
        c.materialize()
        self.assertEqual(c.shape, (2, 2, 2))
        self.assertEqual(c.tolist(), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    #
    # advanced indexing
    #

    def test_advanced_index_1d(self):
        a = ndarray.array([10, 20, 30, 40, 50])
        idx = ndarray.array([2, 0, 4])
        b = a[idx]
        b.materialize()
        self.assertEqual(b.shape, (3,))
        self.assertEqual(b.tolist(), [30, 10, 50])

    def test_advanced_index_1d_repeat(self):
        a = ndarray.array([10, 20, 30])
        idx = ndarray.array([0, 0, 2])
        b = a[idx]
        b.materialize()
        self.assertEqual(b.tolist(), [10, 10, 30])

    def test_advanced_index_1d_from_list(self):
        a = ndarray.array([10, 20, 30, 40])
        b = a[[3, 1]]
        b.materialize()
        self.assertEqual(b.tolist(), [40, 20])

    def test_advanced_index_2d_single_axis(self):
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        idx = ndarray.array([0, 2])
        b = a[idx]
        b.materialize()
        self.assertEqual(b.shape, (2, 3))
        self.assertEqual(b.tolist(), [[1, 2, 3], [7, 8, 9]])

    def test_advanced_index_2d_two_axes(self):
        # a[row, col] → picks a[0,2], a[1,0], a[2,1]
        a = ndarray.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        row = ndarray.array([0, 1, 2])
        col = ndarray.array([2, 0, 1])
        b = a[row, col]
        b.materialize()
        self.assertEqual(b.shape, (3,))
        self.assertEqual(b.tolist(), [3, 4, 8])

    def test_advanced_index_crossentropy_style(self):
        log_probs = ndarray.array([
            [-1.0, -2.0, -3.0],
            [-4.0, -0.5, -6.0],
        ])
        labels = ndarray.array([0, 1])
        result = log_probs[ndarray.arange(2), labels]
        result.materialize()
        self.assertEqual(result.shape, (2,))
        self._assert_list_close(result.tolist(), [-1.0, -0.5])

    #
    # CrossEntropyLoss
    #

    def _cross_entropy(self, logits_batch, labels):
        import math
        n = len(logits_batch)
        total = 0.0
        for logits, label in zip(logits_batch, labels):
            x = [v - max(logits) for v in logits]
            s = math.log(sum(math.exp(v) for v in x))
            total += -(x[label] - s)
        return total / n

    def _ce_forward(self, logits_list, labels_list):
        from mnist import CrossEntropyLoss
        logits = ndarray.array(logits_list)
        labels = ndarray.array(labels_list)
        loss = CrossEntropyLoss().forward(logits, labels)
        loss.materialize()
        return loss.tolist()

    def test_crossentropy_single_sample(self):
        logits = [[2.0, 1.0, 0.1]]
        labels = [0]
        got = self._ce_forward(logits, labels)
        self.assertAlmostEqual(got, self._cross_entropy(logits, labels), places=5)

    def test_crossentropy_batch(self):
        logits = [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]
        labels = [2, 0]
        got = self._ce_forward(logits, labels)
        self.assertAlmostEqual(got, self._cross_entropy(logits, labels), places=5)

    def test_crossentropy_uniform_logits(self):
        # uniform logits → loss = log(C)
        import math
        logits = [[0.0, 0.0, 0.0]]
        labels = [0]
        got = self._ce_forward(logits, labels)
        self.assertAlmostEqual(got, math.log(3), places=5)

    def test_crossentropy_high_confidence_correct(self):
        # 正解クラスに大きなスコア → loss が小さい
        logits = [[10.0, 0.0, 0.0]]
        labels = [0]
        got = self._ce_forward(logits, labels)
        self.assertLess(got, 0.01)

    def test_crossentropy_high_confidence_wrong(self):
        # 間違いクラスに大きなスコア → loss が大きい
        logits = [[10.0, 0.0, 0.0]]
        labels = [1]
        got = self._ce_forward(logits, labels)
        self.assertGreater(got, 9.0)

    def test_crossentropy_numerical_stability(self):
        # 大きな値でも nan/inf にならない
        import math
        logits = [[1000.0, 1001.0, 999.0]]
        labels = [1]
        got = self._ce_forward(logits, labels)
        self.assertFalse(math.isnan(got) or math.isinf(got))
        self.assertAlmostEqual(got, self._cross_entropy(logits, labels), places=5)

if __name__ == '__main__':
    unittest.main()
