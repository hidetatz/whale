package tensor

import (
	"testing"
)

func TestSum(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		keepdims  bool
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar 1",
			tensor:   Scalar(2),
			keepdims: false,
			expected: Scalar(2),
		},
		{
			name:     "scalar 2",
			tensor:   Scalar(2),
			keepdims: true,
			expected: Scalar(2),
		},
		{
			name:      "scalar 3",
			tensor:    Scalar(2),
			keepdims:  true,
			axes:      []int{0},
			expectErr: true,
		},
		{
			name:     "1d 1",
			tensor:   Vector([]float32{1, 2, 3}),
			keepdims: false,
			axes:     []int{},
			expected: Scalar(6),
		},
		{
			name:     "1d 2",
			tensor:   Vector([]float32{1, 2, 3}),
			keepdims: true,
			axes:     []int{},
			expected: Vector([]float32{6}),
		},
		{
			name:     "1d 3",
			tensor:   Vector([]float32{1, 2, 3}),
			keepdims: false,
			axes:     []int{0},
			expected: Scalar(6),
		},
		{
			name:     "1d 4",
			tensor:   Vector([]float32{1, 2, 3}),
			keepdims: true,
			axes:     []int{0},
			expected: Vector([]float32{6}),
		},
		{
			name:      "1d 5",
			tensor:    Vector([]float32{1, 2, 3}),
			keepdims:  true,
			axes:      []int{1},
			expectErr: true,
		},
		{
			name:      "1d 6",
			tensor:    Vector([]float32{1, 2, 3}),
			keepdims:  true,
			axes:      []int{0, 0},
			expectErr: true,
		},
		{
			name:     "2d 1",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: false,
			axes:     []int{},
			expected: Scalar(10),
		},
		{
			name:     "2d 2",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: true,
			axes:     []int{},
			expected: Vector([]float32{10}).Reshape(1, 1),
		},
		{
			name:     "2d 3",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: false,
			axes:     []int{0},
			expected: Vector([]float32{4, 6}),
		},
		{
			name:     "2d 4",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: true,
			axes:     []int{0},
			expected: Vector([]float32{4, 6}).Reshape(1, 2),
		},
		{
			name:     "2d 5",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: false,
			axes:     []int{1},
			expected: Vector([]float32{3, 7}),
		},
		{
			name:     "2d 6",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: true,
			axes:     []int{1},
			expected: Vector([]float32{3, 7}).Reshape(2, 1),
		},
		{
			name:     "2d 7",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: false,
			axes:     []int{0, 1},
			expected: Scalar(10),
		},
		{
			name:     "2d 8",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: false,
			axes:     []int{1, 0},
			expected: Scalar(10),
		},
		{
			name:     "2d 8",
			tensor:   Arange(1, 5, 1).Reshape(2, 2),
			keepdims: true,
			axes:     []int{1, 0},
			expected: Vector([]float32{10}).Reshape(1, 1),
		},
		{
			name:      "2d 9",
			tensor:    Arange(1, 5, 1).Reshape(2, 2),
			keepdims:  true,
			axes:      []int{2},
			expectErr: true,
		},
		{
			name:      "2d 10",
			tensor:    Arange(1, 5, 1).Reshape(2, 2),
			keepdims:  true,
			axes:      []int{1, 1},
			expectErr: true,
		},
		{
			name:     "3d 1",
			tensor:   Arange(1, 9, 1).Reshape(2, 2, 2),
			keepdims: false,
			axes:     []int{},
			expected: Scalar(36),
		},
		{
			name:     "3d 2",
			tensor:   Arange(1, 9, 1).Reshape(2, 2, 2),
			keepdims: true,
			axes:     []int{},
			expected: Vector([]float32{36}).Reshape(1, 1, 1),
		},
		{
			name:     "3d 3",
			tensor:   Arange(1, 9, 1).Reshape(2, 2, 2),
			keepdims: false,
			axes:     []int{0},
			expected: Vector([]float32{6, 8, 10, 12}).Reshape(2, 2),
		},
		{
			name:     "4d 1",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0},
			expected: Vector([]float32{14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}).Reshape(3, 2, 2),
		},
		{
			name:     "4d 2",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{0},
			expected: Vector([]float32{14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}).Reshape(1, 3, 2, 2),
		},
		{
			name:     "4d 3",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{1},
			expected: Vector([]float32{15, 18, 21, 24, 51, 54, 57, 60}).Reshape(2, 2, 2),
		},
		{
			name:     "4d 4",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{2},
			expected: Vector([]float32{4, 6, 12, 14, 20, 22, 28, 30, 36, 38, 44, 46}).Reshape(2, 3, 2),
		},
		{
			name:     "4d 5",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{2},
			expected: Vector([]float32{4, 6, 12, 14, 20, 22, 28, 30, 36, 38, 44, 46}).Reshape(2, 3, 1, 2),
		},
		{
			name:     "4d 6",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{3},
			expected: Vector([]float32{3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}).Reshape(2, 3, 2),
		},
		{
			name:     "4d 7",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{3},
			expected: Vector([]float32{3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}).Reshape(2, 3, 2, 1),
		},
		{
			name:     "4d 8",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0, 1},
			expected: Vector([]float32{66, 72, 78, 84}).Reshape(2, 2),
		},
		{
			name:     "4d 9",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{0, 1},
			expected: Vector([]float32{66, 72, 78, 84}).Reshape(1, 1, 2, 2),
		},
		{
			name:     "4d 10",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0, 2},
			expected: Vector([]float32{32, 36, 48, 52, 64, 68}).Reshape(3, 2),
		},
		{
			name:     "4d 11",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{0, 2},
			expected: Vector([]float32{32, 36, 48, 52, 64, 68}).Reshape(1, 3, 1, 2),
		},
		{
			name:     "4d 12",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0, 3},
			expected: Vector([]float32{30, 38, 46, 54, 62, 70}).Reshape(3, 2),
		},
		{
			name:     "4d 13",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{0, 3},
			expected: Vector([]float32{30, 38, 46, 54, 62, 70}).Reshape(1, 3, 2, 1),
		},
		{
			name:     "4d 14",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0, 1, 2},
			expected: Vector([]float32{144, 156}),
		},
		{
			name:     "4d 15",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{0, 1, 3},
			expected: Vector([]float32{138, 162}),
		},
		{
			name:     "4d 15",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{1, 2, 3},
			expected: Vector([]float32{78, 222}),
		},
		{
			name:     "4d 16",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{1, 2, 3},
			expected: Vector([]float32{78, 222}).Reshape(2, 1, 1, 1),
		},
		{
			name:     "4d 17",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{2, 1, 3},
			expected: Vector([]float32{78, 222}),
		},
		{
			name:     "4d 18",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: false,
			axes:     []int{2, 1, 0, 3},
			expected: Scalar(300),
		},
		{
			name:     "4d 19",
			tensor:   Arange(1, 25, 1).Reshape(2, 3, 2, 2),
			keepdims: true,
			axes:     []int{2, 1, 0, 3},
			expected: Vector([]float32{300}).Reshape(1, 1, 1, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().Sum(tc.keepdims, tc.axes...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestSumTo(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		shape     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "2d 1",
			tensor:   Arange(0, 6, 1).Reshape(2, 3),
			shape:    []int{1, 3},
			expected: NdShape([]float32{3, 5, 7}, 1, 3),
		},
		{
			name:     "2d 2",
			tensor:   Arange(0, 6, 1).Reshape(2, 3),
			shape:    []int{2, 1},
			expected: NdShape([]float32{3, 12}, 2, 1),
		},
		{
			name:     "2d 3",
			tensor:   Arange(0, 6, 1).Reshape(2, 3),
			shape:    []int{1, 1},
			expected: NdShape([]float32{15}, 1, 1),
		},
		{
			name:     "3d 1",
			tensor:   Arange(0, 24, 1).Reshape(2, 3, 4),
			shape:    []int{2, 1, 1},
			expected: NdShape([]float32{66, 210}, 2, 1, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.ErrResponser().SumTo(tc.shape...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestArgmax_Argmin(t *testing.T) {
	tests := []struct {
		name        string
		tensor      *Tensor
		keepdims    bool
		axis        int
		expectErr   bool
		expectedMax *Tensor
		expectedMin *Tensor
	}{
		{
			name:        "scalar",
			tensor:      Scalar(3),
			keepdims:    false,
			axis:        -1,
			expectedMax: Scalar(0),
			expectedMin: Scalar(0),
		},
		{
			name:        "scalar2",
			tensor:      Scalar(3),
			keepdims:    true,
			axis:        -1,
			expectedMax: Scalar(0),
			expectedMin: Scalar(0),
		},
		{
			name:      "scalar error",
			tensor:    Scalar(3),
			keepdims:  true,
			axis:      0,
			expectErr: true,
		},
		{
			name:        "vector",
			tensor:      Vector([]float32{0, 1, 5, 3, 2, 4}),
			keepdims:    false,
			axis:        -1,
			expectedMax: Scalar(2),
			expectedMin: Scalar(0),
		},
		{
			name:        "vector 2",
			tensor:      Vector([]float32{0, 1, 5, 3, 2, 4}),
			keepdims:    false,
			axis:        0,
			expectedMax: Scalar(2),
			expectedMin: Scalar(0),
		},
		{
			name:        "vector 3",
			tensor:      Vector([]float32{0, 1, 5, 3, 2, 4}),
			keepdims:    true,
			axis:        0,
			expectedMax: Vector([]float32{2}),
			expectedMin: Vector([]float32{0}),
		},
		{
			name:      "vector err",
			tensor:    Vector([]float32{0, 1, 5, 3, 2, 4}),
			keepdims:  true,
			axis:      1,
			expectErr: true,
		},
		{
			name: "3d",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    false,
			axis:        -1,
			expectedMax: Scalar(6),
			expectedMin: Scalar(7),
		},
		{
			name: "3d 2",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    true,
			axis:        -1,
			expectedMax: NdShape([]float32{6}, 1, 1, 1),
			expectedMin: NdShape([]float32{7}, 1, 1, 1),
		},
		{
			name: "3d 3",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    false,
			axis:        0,
			expectedMax: NdShape([]float32{1, 0, 1, 1, 0, 0}, 3, 2),
			expectedMin: NdShape([]float32{0, 1, 0, 0, 1, 1}, 3, 2),
		},
		{
			name: "3d 4",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    true,
			axis:        0,
			expectedMax: NdShape([]float32{1, 0, 1, 1, 0, 0}, 1, 3, 2),
			expectedMin: NdShape([]float32{0, 1, 0, 0, 1, 1}, 1, 3, 2),
		},
		{
			name: "3d 5",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    false,
			axis:        1,
			expectedMax: NdShape([]float32{2, 2, 0, 1}, 2, 2),
			expectedMin: NdShape([]float32{0, 0, 2, 0}, 2, 2),
		},
		{
			name: "3d 6",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    true,
			axis:        1,
			expectedMax: NdShape([]float32{2, 2, 0, 1}, 2, 1, 2),
			expectedMin: NdShape([]float32{0, 0, 2, 0}, 2, 1, 2),
		},
		{
			name: "3d 7",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    false,
			axis:        2,
			expectedMax: NdShape([]float32{1, 1, 1, 0, 0, 1}, 2, 3),
			expectedMin: NdShape([]float32{0, 0, 0, 1, 1, 0}, 2, 3),
		},
		{
			name: "3d 8",
			tensor: New([][][]float32{
				{
					{1, 3},
					{2, 7},
					{6, 10},
				},
				{
					{11, 0},
					{9, 8},
					{4, 5},
				},
			}),
			keepdims:    true,
			axis:        2,
			expectedMax: NdShape([]float32{1, 1, 1, 0, 0, 1}, 2, 3, 1),
			expectedMin: NdShape([]float32{0, 0, 0, 1, 1, 0}, 2, 3, 1),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			max, err := tc.tensor.ErrResponser().Argmax(tc.keepdims, tc.axis)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expectedMax, max)

			min, err := tc.tensor.ErrResponser().Argmin(tc.keepdims, tc.axis)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expectedMin, min)
		})
	}
}

func TestTensor_Clip(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		min, max float32
		expected *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(3),
			min:      4,
			max:      5,
			expected: Scalar(4),
		},
		{
			name:     "scalar 2",
			tensor:   Scalar(3),
			min:      1,
			max:      2,
			expected: Scalar(2),
		},
		{
			name:     "vector",
			tensor:   Vector([]float32{1, 2, 3, 4, 5}),
			min:      2,
			max:      4,
			expected: Vector([]float32{2, 2, 3, 4, 4}),
		},
		{
			name: "2d",
			tensor: New([][]float32{
				{1, 2, 3},
				{4, 5, 6},
			}),
			min: 2,
			max: 4,
			expected: New([][]float32{
				{2, 2, 3},
				{4, 4, 4},
			}),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.tensor.Clip(tc.min, tc.max)
			mustEq(t, tc.expected, got)
		})
	}
}
