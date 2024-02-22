package tensor2

import (
	"testing"
)

func TestReshape(t *testing.T) {
	index := func(ten *Tensor, idx ...int) *Tensor {
		ten2, err := ten.Index(idx...)
		if err != nil {
			t.Fatalf("invalid index")
		}
		return ten2
	}
	tests := []struct {
		name      string
		tensor    *Tensor
		shape     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(1),
			shape:    []int{1, 1},
			expected: MustNd([]float64{1}, 1, 1),
		},
		{
			name:     "scalar 2",
			tensor:   Scalar(1),
			shape:    []int{1, 1, 1},
			expected: MustNd([]float64{1}, 1, 1, 1),
		},
		{
			name:      "invalid shape",
			tensor:    Scalar(1),
			shape:     []int{1, 2, 1},
			expectErr: true,
		},
		{
			name:     "vector",
			tensor:   Vector([]float64{1, 2, 3, 4}),
			shape:    []int{1, 4, 1},
			expected: MustNd([]float64{1, 2, 3, 4}, 1, 4, 1),
		},
		{
			name:     "vector 2",
			tensor:   Vector([]float64{1, 2, 3, 4}),
			shape:    []int{2, 2},
			expected: MustNd([]float64{1, 2, 3, 4}, 2, 2),
		},
		{
			name:     "vector after indexed",
			tensor:   index(MustNd([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2), 0),
			shape:    []int{2, 1},
			expected: MustNd([]float64{1, 2}, 2, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Reshape(tc.shape...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}
