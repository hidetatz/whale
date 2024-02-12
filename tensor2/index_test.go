package tensor2

import (
	"testing"
)

func TestIndex(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		indices   []int
		expected  *Tensor
		expectErr bool
	}{
		{
			name:      "scalar cannot indexed",
			tensor:    Scalar(1),
			indices:   []int{},
			expectErr: true,
		},
		{
			name:      "missing index",
			tensor:    Vector([]float64{1, 2, 3}),
			indices:   []int{},
			expectErr: true,
		},
		{
			name:      "too many index",
			tensor:    Vector([]float64{1, 2, 3}),
			indices:   []int{0, 1},
			expectErr: true,
		},
		{
			name:     "vector 1",
			tensor:   Vector([]float64{1, 2, 3}),
			indices:  []int{0},
			expected: Scalar(1),
		},
		{
			name:     "2d 1",
			tensor:   MustNd([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			indices:  []int{0},
			expected: Vector([]float64{1, 2, 3}),
		},
		{
			name:     "2d 2",
			tensor:   MustNd([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			indices:  []int{0, 1},
			expected: Scalar(2),
		},
		{
			name:      "too many index",
			tensor:    MustNd([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			indices:   []int{0, 1, 1},
			expectErr: true,
		},
		{
			name:      "too big index",
			tensor:    MustNd([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			indices:   []int{0, 3},
			expectErr: true,
		},
		{
			name:     "containing 0 in stride 1",
			tensor:   &Tensor{data: seqf(1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			indices:  []int{0},
			expected: MustNd(seqf(1, 25), 2, 3, 4),
		},
		{
			name:     "containing 0 in stride 2",
			tensor:   &Tensor{data: seqf(1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			indices:  []int{1},
			expected: MustNd(seqf(1, 25), 2, 3, 4),
		},
		{
			name:     "containing 0 in stride 3",
			tensor:   &Tensor{data: seqf(1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			indices:  []int{1, 1},
			expected: MustNd(seqf(13, 25), 3, 4),
		},
		{
			name:     "containing 0 in stride 4",
			tensor:   &Tensor{data: seqf(1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			indices:  []int{1, 1, 2},
			expected: MustNd(seqf(21, 25), 4),
		},
		{
			name:     "containing 0 in stride 5",
			tensor:   &Tensor{data: seqf(1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			indices:  []int{1, 1, 2, 3},
			expected: Scalar(24),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Index(tc.indices...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestIndex_Complicated(t *testing.T) {
	tensor := MustNd(seqf(1, 121), 2, 3, 4, 5)

	tensor2, err := tensor.Index(0)
	checkErr(t, false, err)
	mustEq(t, MustNd(seqf(1, 61), 3, 4, 5), tensor2)

	tensor3, err := tensor2.Index(2)
	checkErr(t, false, err)
	mustEq(t, MustNd(seqf(41, 61), 4, 5), tensor3)

	tensor4, err := tensor3.Index(1)
	checkErr(t, false, err)
	mustEq(t, Vector(seqf(46, 51)), tensor4)

	tensor5, err := tensor4.Index(4)
	checkErr(t, false, err)
	mustEq(t, Scalar(50), tensor5)
}
