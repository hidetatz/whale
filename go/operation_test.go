package whale

import (
	"reflect"
	"slices"
	"testing"

	"github.com/hidetatz/whale/tensor"
)

// Tests each calculation function correctness.
// Detailed operation test should be done in tensor/device tests.
// This test should focus on the correctness of forward/backward implementation.
func TestSingleOperations(t *testing.T) {
	tests := []struct {
		name string

		fn any
		in []*tensor.Tensor
		// extra arguments to the fn
		extra []any
		// flag if the last param of arguments is variable length
		variable bool

		expected     []*tensor.Tensor
		expectedGrad []*tensor.Tensor
	}{
		{
			name:         "reshape",
			fn:           Reshape,
			in:           ts(arng(t, 1, 13, 2, 2, 3)),
			extra:        []any{[]int{6, 2}},
			variable:     true,
			expected:     ts(arng(t, 1, 13, 6, 2)),
			expectedGrad: ts(tensor.Ones(2, 2, 3)),
		},
		{
			name:         "transpose",
			fn:           Transpose,
			in:           ts(arng(t, 1, 13, 2, 2, 3)),
			expected:     ts(arng(t, 1, 13, 2, 2, 3).Transpose()),
			expectedGrad: ts(tensor.Ones(2, 2, 3)),
		},
		{
			name:         "broadcastto",
			fn:           BroadcastTo,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{[]int{3, 2, 3}},
			variable:     true,
			expected:     ts(tensor.NdShape([]float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, 3, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{3, 3, 3, 3, 3, 3}, 2, 3)),
		},
		{
			name:         "sum",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{false, []int{0}},
			variable:     true,
			expected:     ts(tensor.NdShape([]float32{5, 7, 9}, 3)),
			expectedGrad: ts(tensor.Ones(2, 3)),
		},
		{
			name:         "sum2",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{false, []int{1}},
			variable:     true,
			expected:     ts(tensor.NdShape([]float32{6, 15}, 2)),
			expectedGrad: ts(tensor.Ones(2, 3)),
		},
		{
			name:         "sum3",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{true, []int{1}},
			variable:     true,
			expected:     ts(tensor.NdShape([]float32{6, 15}, 2, 1)),
			expectedGrad: ts(tensor.Ones(2, 3)),
		},
		{
			name:         "sumTo",
			fn:           SumTo,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{[]int{1, 3}},
			variable:     true,
			expected:     ts(tensor.NdShape([]float32{5, 7, 9}, 1, 3)),
			expectedGrad: ts(tensor.Ones(2, 3)),
		},
		{
			name:         "exp",
			fn:           Exp,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766, 403.4287934927351}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766, 403.4287934927351}, 2, 3)),
		},
		{
			name:         "add",
			fn:           Add,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{8, 10, 12, 14, 16, 18}, 2, 3)),
			expectedGrad: ts(tensor.Ones(2, 3), tensor.Ones(2, 3)),
		},
		{
			name:         "add2",
			fn:           Add,
			in:           ts(arng(t, 1, 7, 2, 3), tensor.Scalar(2)),
			expected:     ts(tensor.NdShape([]float32{3, 4, 5, 6, 7, 8}, 2, 3)),
			expectedGrad: ts(tensor.Ones(2, 3), tensor.Scalar(6)),
		},
		{
			name:         "sub",
			fn:           Sub,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{-6, -6, -6, -6, -6, -6}, 2, 3)),
			expectedGrad: ts(tensor.Ones(2, 3), tensor.NdShape([]float32{-1, -1, -1, -1, -1, -1}, 2, 3)),
		},
		{
			name:         "sub2",
			fn:           Sub,
			in:           ts(arng(t, 1, 7, 2, 3), tensor.Scalar(2)),
			expected:     ts(tensor.NdShape([]float32{-1, 0, 1, 2, 3, 4}, 2, 3)),
			expectedGrad: ts(tensor.Ones(2, 3), tensor.Scalar(-6)),
		},
		{
			name:         "mul",
			fn:           Mul,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{7, 16, 27, 40, 55, 72}, 2, 3)),
			expectedGrad: ts(arng(t, 7, 13, 2, 3), arng(t, 1, 7, 2, 3)),
		},
		{
			name:         "mul2",
			fn:           Mul,
			in:           ts(arng(t, 1, 7, 2, 3), tensor.Scalar(2)),
			expected:     ts(tensor.NdShape([]float32{2, 4, 6, 8, 10, 12}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{2, 2, 2, 2, 2, 2}, 2, 3), tensor.Scalar(21)),
		},
		{
			name: "div",
			fn:   Div,
			in: ts(
				tensor.NdShape([]float32{2, 4, 6, 8, 10, 12}, 2, 3),
				tensor.NdShape([]float32{1, 2, 3, 4, 5, 6}, 2, 3),
			),
			expected: ts(
				tensor.NdShape([]float32{2, 2, 2, 2, 2, 2}, 2, 3)),
			expectedGrad: ts(
				tensor.NdShape([]float32{1, 0.5, 0.3333333333333333, 0.25, 0.2, 0.166666666666666666}, 2, 3),
				tensor.NdShape([]float32{-2.0, -1.0, -0.6666666666666666, -0.5, -0.4, -0.3333333333333333}, 2, 3),
			),
		},
		{
			name:         "div2",
			fn:           Div,
			in:           ts(arng(t, 1, 7, 2, 3), tensor.Scalar(2)),
			expected:     ts(tensor.NdShape([]float32{0.5, 1, 1.5, 2, 2.5, 3}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 2, 3), tensor.Scalar(-5.25)),
		},
		{
			name:         "neg",
			fn:           Neg,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{-1, -2, -3, -4, -5, -6}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{-1, -1, -1, -1, -1, -1}, 2, 3)),
		},
		{
			name:         "pow",
			fn:           Pow,
			in:           ts(arng(t, 1, 7, 2, 3), tensor.Scalar(2)),
			expected:     ts(tensor.NdShape([]float32{1, 4, 9, 16, 25, 36}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{2, 4, 6, 8, 10, 12}, 2, 3), nil),
		},
		{
			name:         "sin",
			fn:           Sin,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{0.8414709848078965, 0.9092974268256816, 0.1411200080598672, -0.7568024953079282, -0.9589242746631385, -0.27941549819892586}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{0.5403023058681398, -0.4161468365471424, -0.9899924966004454, -0.6536436208636119, 0.2836621854632263, 0.9601702866503661}, 2, 3)),
		},
		{
			name:         "cos",
			fn:           Cos,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{0.5403023058681398, -0.4161468365471424, -0.9899924966004454, -0.6536436208636119, 0.2836621854632263, 0.9601702866503661}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{-0.8414709848078965, -0.9092974268256816, -0.1411200080598672, 0.7568024953079282, 0.9589242746631385, 0.27941549819892586}, 2, 3)),
		},
		{
			name:         "tanh",
			fn:           Tanh,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(tensor.NdShape([]float32{0.7615941559557649, 0.9640275800758169, 0.9950547536867305, 0.999329299739067, 0.9999092042625951, 0.9999877116507956}, 2, 3)),
			expectedGrad: ts(tensor.NdShape([]float32{0.41997433, 0.070650816, 0.009865999, 0.0013408661, 0.00018155575, 2.4557114e-05}, 2, 3)),
		},
		{
			name:         "matmul",
			fn:           MatMul,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 1, 7, 3, 2)),
			expected:     ts(tensor.NdShape([]float32{22, 28, 49, 64}, 2, 2)),
			expectedGrad: ts(tensor.NdShape([]float32{3, 7, 11, 3, 7, 11}, 2, 3), tensor.NdShape([]float32{5, 5, 7, 7, 9, 9}, 3, 2)),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			xs := make([]*Variable, len(tc.in))
			args := []reflect.Value{}
			for i, in := range tc.in {
				v := NewVar(in)
				xs[i] = v
				args = append(args, reflect.ValueOf(v))
			}

			for _, e := range tc.extra {
				args = append(args, reflect.ValueOf(e))
			}

			f := reflect.ValueOf(tc.fn)

			var results []reflect.Value
			if tc.variable {
				results = f.CallSlice(args)
			} else {
				results = f.Call(args)
			}

			if len(results) != 2 {
				t.Errorf("unexpected return length: %v", len(results))
			}

			// check err
			if !results[1].IsNil() {
				err, ok := results[1].Interface().(error)
				if !ok {
					t.Errorf("unexpected second-returned value: %v", results[1])
				}

				if err != nil {
					t.Errorf("unexpected err on forward: %v", err)
				}
			}

			// check returned
			y, ok := results[0].Interface().(*Variable)
			if !ok {
				t.Errorf("unexpected first-returned value: %v", results[0])
			}

			verify(t, xs, vs(y), tc.expected, tc.expectedGrad)

			// if tc.dezeroX != nil && tc.dezeroY != nil {
			// 	verifyDezeroS(t, tc.name, tc.dezeroX, tc.dezeroY, tc.expected, tc.expectedGrad)
			// }
		})
	}
}

// do this:
//   - check output is the same as expected
//   - call Backward() for each output
//   - check gradient of input is the same as expected
func verify(t *testing.T, in, out []*Variable, expected, expectedGrad []*tensor.Tensor) {
	t.Helper()

	// check calc output
	if len(expected) != len(out) {
		t.Errorf("result length mismatch expected %v but got %v", len(expected), len(out))
	}

	for i := range expected {
		if !expected[i].Equals(out[i].data) {
			t.Errorf("expected %v but got %v", expected[i], out[i].data)
		}
	}

	// check gradient
	for _, o := range out {
		if err := o.Backward(); err != nil {
			t.Errorf("unexpected err on backward: %v", err)
		}
	}

	if len(expectedGrad) != len(in) {
		t.Errorf("grad length mismatch expected %v but got %v", len(expectedGrad), len(in))
	}

	for i := range expectedGrad {
		// Comes here on Pow test.
		// In Pow, the gradient of x is calculated, but c is not. That's why
		// this check is needed.
		if expectedGrad[i] == nil {
			if in[i].grad != nil {
				t.Errorf("grad: expected nil but got %v", in[i].grad.data)
			}
		} else {
			if !expectedGrad[i].Equals(in[i].grad.data) {
				t.Errorf("grad: expected %v but got %v", expectedGrad[i], in[i].grad.data)
			}
		}
	}
}

func check[S ~[]E, E comparable](t *testing.T, expected, got S, name string) {
	if !slices.Equal(expected, got) {
		t.Fatalf("mismatch with python output (%s): expected: %v, got: %v", name, expected, got)
	}
}

/*
 * tensor factory helpers
 */

func arng(t *testing.T, from, to int, shape ...int) *tensor.Tensor {
	t.Helper()
	return tensor.Arange(float32(from), float32(to), 1).Reshape(shape...)
}

func ones(t *testing.T, shape ...int) *tensor.Tensor {
	t.Helper()
	return tensor.Ones(shape...)
}

func trans(t *testing.T, orig *tensor.Tensor, axes ...int) *tensor.Tensor {
	t.Helper()
	return orig.Transpose(axes...)
}

// makes slice of tensors
func ts(tensors ...*tensor.Tensor) []*tensor.Tensor {
	return tensors
}

// makes slice of variable
func vs(vs ...*Variable) []*Variable {
	return vs
}
