package whale

import (
	"testing"

	"github.com/hidetatz/whale/tensor"
)

func arng(t *testing.T, from, to int, shape ...int) *tensor.Tensor {
	t.Helper()
	tsr, _ := tensor.ArangeFrom(from, to).Reshape(shape...)
	return tsr
}

func ones(t *testing.T, shape ...int) *tensor.Tensor {
	t.Helper()
	return tensor.Ones(shape...)
}

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

// makes slice of tensors
func ts(tensors ...*tensor.Tensor) []*tensor.Tensor {
	return tensors
}

// makes slice of variable
func vs(vs ...*Variable) []*Variable {
	return vs
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name     string
		in       *tensor.Tensor
		shape    []int

		// verification
		expected *tensor.Tensor
		grad *tensor.Tensor

		// dezero verification
		dezero string
	}{
		{
			name:     "reshape",
			in:       arng(t, 1, 13, 2, 2, 3),
			shape:    []int{6, 2},
			expected: arng(t, 1, 13, 6, 2),
			grad:     ones(t, 2, 2, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			x := NewVar(tc.in)
			y, err := Reshape(x, tc.shape...)
			if err != nil {
				t.Errorf("unexpected err on forward: %v", err)
			}

			verify(t, vs(x), vs(y), ts(tc.expected), ts(tc.grad))
		})
	}
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name     string
		in       *tensor.Tensor

		// verification
		expected *tensor.Tensor
		grad *tensor.Tensor

		// dezero verification
		dezero string
	}{
		{
			name:     "reshape",
			in:       arng(t, 1, 13, 2, 2, 3),
			expected: arng(t, 1, 13, 2, 2, 3).Transpose(),
			grad:     ones(t, 2, 2, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			t.Log(tc.in.RawString())
			t.Log(tc.expected.RawString())
			t.Log(tc.grad.RawString())

			x := NewVar(tc.in)
			y, err := Transpose(x)
			if err != nil {
				t.Errorf("unexpected err on forward: %v", err)
			}

			verify(t, vs(x), vs(y), ts(tc.expected), ts(tc.grad))
		})
	}
}
