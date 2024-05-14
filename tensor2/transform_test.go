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
			expected: NdShape([]float64{1}, 1, 1),
		},
		{
			name:     "scalar 2",
			tensor:   Scalar(1),
			shape:    []int{1, 1, 1},
			expected: NdShape([]float64{1}, 1, 1, 1),
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
			expected: NdShape([]float64{1, 2, 3, 4}, 1, 4, 1),
		},
		{
			name:     "vector 2",
			tensor:   Vector([]float64{1, 2, 3, 4}),
			shape:    []int{2, 2},
			expected: NdShape([]float64{1, 2, 3, 4}, 2, 2),
		},
		{
			name:     "vector after indexed",
			tensor:   NdShape([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2).Index(At(0)),
			shape:    []int{2, 1},
			expected: NdShape([]float64{1, 2}, 2, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().Reshape(tc.shape...)
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
			tensor: ArangeVec(1, 25, 1).Reshape(2, 3, 4),
			axes:   []int{},
			expected: New([][][]float64{
				{
					{1, 13},
					{5, 17},
					{9, 21},
				},
				{
					{2, 14},
					{6, 18},
					{10, 22},
				},
				{
					{3, 15},
					{7, 19},
					{11, 23},
				},
				{
					{4, 16},
					{8, 20},
					{12, 24},
				},
			}),
		},
		{
			name:   "3d 2",
			tensor: ArangeVec(1, 25, 1).Reshape(2, 3, 4),
			axes:   []int{2, 1, 0},
			expected: New([][][]float64{
				{
					{1, 13},
					{5, 17},
					{9, 21},
				},
				{
					{2, 14},
					{6, 18},
					{10, 22},
				},
				{
					{3, 15},
					{7, 19},
					{11, 23},
				},
				{
					{4, 16},
					{8, 20},
					{12, 24},
				},
			}),
		},
		{
			name:   "3d 3",
			tensor: ArangeVec(1, 25, 1).Reshape(2, 3, 4),
			axes:   []int{0, 2, 1},
			expected: New([][][]float64{
				{
					{1, 5, 9},
					{2, 6, 10},
					{3, 7, 11},
					{4, 8, 12},
				},
				{
					{13, 17, 21},
					{14, 18, 22},
					{15, 19, 23},
					{16, 20, 24},
				},
			}),
		},
		{
			name:      "non-unique axes",
			tensor:    ArangeVec(1, 25, 1).Reshape(2, 3, 4),
			axes:      []int{0, 2, 0},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().Transpose(tc.axes...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestSqueeze(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "2d 1",
			tensor:   ArangeVec(0, 6, 1).Reshape(2, 3),
			axes:     []int{},
			expected: ArangeVec(0, 6, 1).Reshape(2, 3),
		},
		{
			name:     "3d 1",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 2, 3),
			axes:     []int{},
			expected: ArangeVec(0, 6, 1).Reshape(2, 3),
		},
		{
			name:     "3d 2",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 6, 1),
			axes:     []int{},
			expected: ArangeVec(0, 6, 1),
		},
		{
			name:     "3d 3",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 6, 1),
			axes:     []int{0},
			expected: ArangeVec(0, 6, 1).Reshape(6, 1),
		},
		{
			name:     "3d 4",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 6, 1),
			axes:     []int{2},
			expected: ArangeVec(0, 6, 1).Reshape(1, 6),
		},
		{
			name:     "3d 5",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 6, 1),
			axes:     []int{0, 2},
			expected: ArangeVec(0, 6, 1),
		},
		{
			name:     "3d 6",
			tensor:   Scalar(3).Reshape(1, 1, 1),
			axes:     []int{0, 2},
			expected: Vector([]float64{3}),
		},
		{
			name:     "3d 6",
			tensor:   Scalar(3).Reshape(1, 1, 1),
			axes:     []int{0, 2, 1},
			expected: Scalar(3),
		},
		{
			name:     "offset non-0",
			tensor:   ArangeVec(0, 6, 1).Reshape(1, 6, 1).Index(All(), FromTo(2, 5), All()),
			axes:     []int{},
			expected: ArangeVec(2, 5, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().Squeeze(tc.axes...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestBroadcastTo(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		shape     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(3),
			shape:    []int{3},
			expected: Vector([]float64{3, 3, 3}),
		},
		{
			name:     "scalar 2",
			tensor:   Scalar(3),
			shape:    []int{3, 4},
			expected: Full(3, 3, 4),
		},
		{
			name:      "vector 1",
			tensor:    Vector([]float64{1, 2, 3}),
			shape:     []int{4},
			expectErr: true,
		},
		{
			name:     "vector 2",
			tensor:   Vector([]float64{1, 2, 3}),
			shape:    []int{3, 3},
			expected: Vector([]float64{1, 2, 3, 1, 2, 3, 1, 2, 3}).Reshape(3, 3),
		},
		{
			name:     "2d 1",
			tensor:   ArangeVec(1, 7, 1).Reshape(2, 3),
			shape:    []int{2, 2, 3},
			expected: Vector([]float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}).Reshape(2, 2, 3),
		},
		{
			name:      "2d err1",
			tensor:    ArangeVec(1, 7, 1).Reshape(2, 3),
			shape:     []int{2, 4, 3},
			expectErr: true,
		},
		{
			name:     "2d 2",
			tensor:   ArangeVec(1, 7, 1).Reshape(1, 6),
			shape:    []int{2, 2, 6},
			expected: Vector([]float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}).Reshape(2, 2, 6),
		},
		{
			name:      "2d err",
			tensor:    ArangeVec(1, 7, 1).Reshape(1, 6),
			shape:     []int{2, 2, 12},
			expectErr: true,
		},
		{
			name:   "indexed",
			tensor: ArangeVec(0, 24, 1).Reshape(2, 1, 12).Index(At(1)),
			shape:  []int{2, 2, 12},
			expected: New([][][]float64{
				{
					{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
					{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
				},
				{
					{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
					{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
				},
			}),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().BroadcastTo(tc.shape...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}
