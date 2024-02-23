package tensor2

import (
	"testing"
)

func TestReshape(t *testing.T) {
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
			tensor:   Must(MustNd([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2).Index(0)),
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

func TestTranspose(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(1),
			axes:     []int{},
			expected: Scalar(1),
		},
		{
			name:     "scalar",
			tensor:   Scalar(1),
			axes:     []int{0},
			expected: Scalar(1),
		},
		{
			name:     "vector",
			tensor:   ArangeVec(1, 13, 1),
			axes:     []int{},
			expected: ArangeVec(1, 13, 1),
		},
		{
			name:     "vector",
			tensor:   ArangeVec(1, 13, 1),
			axes:     []int{0},
			expected: ArangeVec(1, 13, 1),
		},
		{
			name:      "axes too big",
			tensor:    ArangeVec(1, 13, 1),
			axes:      []int{1},
			expectErr: true,
		},
		{
			name:      "axes too many",
			tensor:    ArangeVec(1, 13, 1),
			axes:      []int{0, 1},
			expectErr: true,
		},
		{
			name:   "3d",
			tensor: Must(ArangeVec(1, 25, 1).Reshape(2, 3, 4)),
			axes:   []int{},
			expected: MustNd([]float64{
				1, 13, 5, 17, 9, 21,
				2, 14, 6, 18, 10, 22,
				3, 15, 7, 19, 11, 23,
				4, 16, 8, 20, 12, 24,
			}, 4, 3, 2),
		},
		{
			name:   "3d 2",
			tensor: Must(ArangeVec(1, 25, 1).Reshape(2, 3, 4)),
			axes:   []int{2, 1, 0},
			expected: MustNd([]float64{
				1, 13, 5, 17, 9, 21,
				2, 14, 6, 18, 10, 22,
				3, 15, 7, 19, 11, 23,
				4, 16, 8, 20, 12, 24,
			}, 4, 3, 2),
		},
		{
			name:   "3d 3",
			tensor: Must(ArangeVec(1, 25, 1).Reshape(2, 3, 4)),
			axes:   []int{0, 2, 1},
			expected: MustNd([]float64{
				1, 5, 9,
				2, 6, 10,
				3, 7, 11,
				4, 8, 12,
				13, 17, 21,
				14, 18, 22,
				15, 19, 23,
				16, 20, 24,
			}, 2, 4, 3),
		},
		{
			name:      "non-unique axes",
			tensor:    Must(ArangeVec(1, 25, 1).Reshape(2, 3, 4)),
			axes:      []int{0, 2, 0},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Transpose(tc.axes...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}
