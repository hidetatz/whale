package whale

import (
	"reflect"
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
		in *tensor.Tensor
		// extra arguments to the fn
		extra []any
		// flag if the last param of arguments is variable length
		variable bool

		expected     *tensor.Tensor
		expectedGrad *tensor.Tensor

		// dezero verification
		dezero string
	}{
		{
			name:         "reshape",
			fn:           Reshape,
			in:           arng(t, 1, 13, 2, 2, 3),
			extra:        []any{[]int{6, 2}},
			variable:     true,
			expected:     arng(t, 1, 13, 6, 2),
			expectedGrad: ones(t, 2, 2, 3),
		},
		{
			name:         "transpose",
			fn:           Transpose,
			in:           arng(t, 1, 13, 2, 2, 3),
			expected:     arng(t, 1, 13, 2, 2, 3).Transpose(),
			expectedGrad: ones(t, 2, 2, 3),
		},
		{
			name:         "broadcastto",
			fn:           BroadcastTo,
			in:           arng(t, 1, 7, 2, 3),
			extra:        []any{[]int{3, 2, 3}},
			variable:     true,
			expected:     nd(t, []float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, 3, 2, 3),
			expectedGrad: nd(t, []float64{3, 3, 3, 3, 3, 3}, 2, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			x := NewVar(tc.in)

			f := reflect.ValueOf(tc.fn)

			args := []reflect.Value{reflect.ValueOf(x)}
			for _, e := range tc.extra {
				args = append(args, reflect.ValueOf(e))
			}

			var results []reflect.Value
			if tc.variable {
				results = f.CallSlice(args)
			} else {
				results = f.Call(args)
			}

			if len(results) != 2 {
				t.Errorf("unexpected return length: %v", len(results))
			}

			y, ok := results[0].Interface().(*Variable)
			if !ok {
				t.Errorf("unexpected first-returned value: %v", results[0])
			}

			if !results[1].IsNil() {
				err, ok := results[1].Interface().(error)
				if !ok {
					t.Errorf("unexpected second-returned value: %v", results[1])
				}

				if err != nil {
					t.Errorf("unexpected err on forward: %v", err)
				}
			}

			verify(t, vs(x), vs(y), ts(tc.expected), ts(tc.expectedGrad))
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
		if !expectedGrad[i].Equals(in[i].grad.data) {
			t.Errorf("grad: expected %v but got %v", expectedGrad[i].RawString(), in[i].grad.data.RawString())
		}
	}
}

/*
 * tensor factory helpers
 */
func nd(t *testing.T, data []float64, shape ...int) *tensor.Tensor {
	t.Helper()
	tsr, _ := tensor.Nd(data, shape...)
	return tsr
}

func arng(t *testing.T, from, to int, shape ...int) *tensor.Tensor {
	t.Helper()
	tsr, _ := tensor.ArangeFrom(from, to).Reshape(shape...)
	return tsr
}

func ones(t *testing.T, shape ...int) *tensor.Tensor {
	t.Helper()
	return tensor.Ones(shape...)
}

// makes slice of tensors
func ts(tensors ...*tensor.Tensor) []*tensor.Tensor {
	return tensors
}

// makes slice of variable
func vs(vs ...*Variable) []*Variable {
	return vs
}
