package tensor

import (
	"testing"
)

func mustEq(t *testing.T, expected, got *Tensor) {
	t.Helper()
	if expected == nil {
		if got != nil {
			t.Fatalf("expected nil but got %v", got)
		}
		return
	}
	if !expected.Equals(got) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func checkErr(t *testing.T, expectErr bool, err error) {
	t.Helper()
	if (err != nil) != expectErr {
		t.Fatalf("unexpected error: expected: [%v] but got [%v]", expectErr, err)
	}
}

func TestScalar(t *testing.T) {
	got := Scalar(0.1)
	mustEq(t, &Tensor{data: []float64{0.1}}, got)
}

func TestVector(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		shape    int
		expected *Tensor
	}{
		{
			name:     "simple",
			data:     []float64{1, 2, 3},
			expected: &Tensor{data: []float64{1, 2, 3}, offset: 0, Shape: []int{3}, Strides: []int{1}},
		},
		{
			name:     "empty",
			data:     []float64{},
			expected: &Tensor{data: []float64{}, offset: 0, Shape: []int{0}, Strides: []int{1}},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := Vector(tc.data)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestNdShape(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			data:     []float64{1},
			shape:    []int{},
			expected: Scalar(1),
		},
		{
			name:     "vector1",
			data:     []float64{1, 2, 3},
			shape:    []int{},
			expected: Vector([]float64{1, 2, 3}),
		},
		{
			name:     "vector2",
			data:     []float64{1},
			shape:    []int{1},
			expected: Vector([]float64{1}),
		},
		{
			name:      "vector err",
			data:      []float64{1, 2, 3},
			shape:     []int{2},
			expectErr: true,
		},
		{
			name:     "matrix1",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{2, 2},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{2, 2}, Strides: []int{2, 1}},
		},
		{
			name:     "matrix2",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{1, 4},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{1, 4}, Strides: []int{4, 1}},
		},
		{
			name:     "matrix3",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{4, 1},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{4, 1}, Strides: []int{1, 1}},
		},
		{
			name:      "matrix err",
			data:      []float64{1, 2, 3, 4},
			shape:     []int{4, 2},
			expectErr: true,
		},
		{
			name:     "3d",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{4, 1, 1},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{4, 1, 1}, Strides: []int{1, 1, 1}},
		},
		{
			name:     "3d2",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{2, 2, 1},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{2, 2, 1}, Strides: []int{2, 1, 1}},
		},
		{
			name:     "4d",
			data:     []float64{1, 2, 3, 4},
			shape:    []int{1, 1, 1, 4},
			expected: &Tensor{data: []float64{1, 2, 3, 4}, offset: 0, Shape: []int{1, 1, 1, 4}, Strides: []int{4, 4, 4, 1}},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := RespErr.NdShape(tc.data, tc.shape...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestNew(t *testing.T) {
	tests := []struct {
		name      string
		arr       any
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			arr:      1,
			expected: Scalar(1),
		},
		{
			name:     "vector",
			arr:      []float64{1, 2, 3, 4, 5},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:      "neither int nor float",
			arr:       []string{"a"},
			expectErr: true,
		},
		{
			name: "2d 1",
			arr: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8, 9},
			},
			expected: Arange(1, 10, 1).Reshape(3, 3),
		},
		{
			name: "2d 2",
			arr: [][]float64{
				{},
				{},
				{},
			},
			expected: Vector([]float64{}).Reshape(3, 0),
		},
		{
			name: "non homogeneous",
			arr: [][]float64{
				{1, 2, 3},
				{4, 5, 6},
				{7, 8},
			},
			expectErr: true,
		},
		{
			name: "3d",
			arr: [][][]float64{
				{
					{1, 2, 3},
					{4, 5, 6},
					{7, 8, 9},
				},
				{
					{10, 11, 12},
					{13, 14, 15},
					{16, 17, 18},
				},
				{
					{19, 20, 21},
					{22, 23, 24},
					{25, 26, 27},
				},
			},
			expected: Arange(1, 28, 1).Reshape(3, 3, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := RespErr.New(tc.arr)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestFactories(t *testing.T) {
	tests := []struct {
		name     string
		factory  func() *Tensor
		expected *Tensor
	}{
		{
			name: "zeros",
			factory: func() *Tensor {
				return Zeros(2, 2, 2)
			},
			expected: &Tensor{
				data:    []float64{0, 0, 0, 0, 0, 0, 0, 0},
				offset:  0,
				Shape:   []int{2, 2, 2},
				Strides: []int{4, 2, 1},
			},
		},
		{
			name: "zeroslike",
			factory: func() *Tensor {
				return ZerosLike(NdShape([]float64{1, 2, 3, 4}, 2, 2))
			},
			expected: &Tensor{
				data:    []float64{0, 0, 0, 0},
				offset:  0,
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
		},
		{
			name: "ones",
			factory: func() *Tensor {
				return Ones(2, 2, 2)
			},
			expected: &Tensor{
				data:    []float64{1, 1, 1, 1, 1, 1, 1, 1},
				offset:  0,
				Shape:   []int{2, 2, 2},
				Strides: []int{4, 2, 1},
			},
		},
		{
			name: "oneslike",
			factory: func() *Tensor {
				return OnesLike(NdShape([]float64{1, 2, 3, 4}, 2, 2))
			},
			expected: &Tensor{
				data:    []float64{1, 1, 1, 1},
				offset:  0,
				Shape:   []int{2, 2},
				Strides: []int{2, 1},
			},
		},
		{
			name: "full",
			factory: func() *Tensor {
				return Full(3, 2, 2, 2)
			},
			expected: &Tensor{
				data:    []float64{3, 3, 3, 3, 3, 3, 3, 3},
				offset:  0,
				Shape:   []int{2, 2, 2},
				Strides: []int{4, 2, 1},
			},
		},
		{
			name: "arange",
			factory: func() *Tensor {
				return Arange(0, 8, 1)
			},
			expected: &Tensor{
				data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				offset:  0,
				Shape:   []int{8},
				Strides: []int{1},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.factory()
			mustEq(t, tc.expected, got)
		})
	}
}
