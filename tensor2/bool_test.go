package tensor2

import "testing"

func TestBool(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		f        func(f float64) bool
		expected *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(3),
			f:        func(f float64) bool { return f < 0 },
			expected: Scalar(0),
		},
		{
			name:     "vector",
			tensor:   Arange(1, 4, 1),
			f:        func(f float64) bool { return f < 2 },
			expected: Vector([]float64{1, 0, 0}),
		},
		{
			name:     "2d",
			tensor:   Arange(1, 7, 1).Reshape(2, 3),
			f:        func(f float64) bool { return int(f)%2 == 0 },
			expected: Vector([]float64{0, 1, 0, 1, 0, 1}).Reshape(2, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.tensor.Bool(tc.f)
			mustEq(t, tc.expected, got)
		})
	}
}
