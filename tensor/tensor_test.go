package tensor

import (
	"slices"
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
		t.Errorf("unexpected error: expected: %v but got %v", expectErr, err)
	}
}

func TestEmpty(t *testing.T) {
	got := Empty()
	mustEq(t, &Tensor{}, got)
}

func TestScalar(t *testing.T) {
	got := Scalar(0.1)
	mustEq(t, &Tensor{Data: []float64{0.1}}, got)
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
			expected: &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
		},
		{
			name:     "empty",
			data:     []float64{},
			expected: &Tensor{Data: []float64{}, Shape: []int{0}},
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

func TestNd(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "vector1",
			data:  []float64{1},
			shape: []int{1},
			expected: &Tensor{
				Data:  []float64{1},
				Shape: []int{1},
			},
		},
		{
			name:  "vector2",
			data:  []float64{1, 2, 3},
			shape: []int{3},
			expected: &Tensor{
				Data:  []float64{1, 2, 3},
				Shape: []int{3},
			},
		},
		{
			name:      "vector err",
			data:      []float64{1, 2, 3},
			shape:     []int{2},
			expectErr: true,
		},
		{
			name:  "matrix1",
			data:  []float64{1, 2, 3, 4},
			shape: []int{2, 2},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{2, 2},
			},
		},
		{
			name:  "matrix2",
			data:  []float64{1, 2, 3, 4},
			shape: []int{1, 4},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{1, 4},
			},
		},
		{
			name:  "matrix3",
			data:  []float64{1, 2, 3, 4},
			shape: []int{4, 1},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{4, 1},
			},
		},
		{
			name:      "matrix err",
			data:      []float64{1, 2, 3, 4},
			shape:     []int{4, 2},
			expectErr: true,
		},
		{
			name:  "3d",
			data:  []float64{1, 2, 3, 4},
			shape: []int{4, 1, 1},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{4, 1, 1},
			},
		},
		{
			name:  "3d2",
			data:  []float64{1, 2, 3, 4},
			shape: []int{2, 2, 1},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{2, 2, 1},
			},
		},
		{
			name:  "4d",
			data:  []float64{1, 2, 3, 4},
			shape: []int{1, 1, 1, 4},
			expected: &Tensor{
				Data:  []float64{1, 2, 3, 4},
				Shape: []int{1, 1, 1, 4},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := Nd(tc.data, tc.shape...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestFactories(t *testing.T) {
	tests := []struct {
		name      string
		factory   func() (*Tensor, error)
		expectErr bool
		expected  *Tensor
	}{
		{
			name: "zeros",
			factory: func() (*Tensor, error) {
				return Zeros(2, 2, 2), nil
			},
			expected: &Tensor{
				Data:  []float64{0, 0, 0, 0, 0, 0, 0, 0},
				Shape: []int{2, 2, 2},
			},
		},
		{
			name: "zeroslike",
			factory: func() (*Tensor, error) {
				return ZerosLike(&Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}}), nil
			},
			expected: &Tensor{
				Data:  []float64{0, 0, 0, 0},
				Shape: []int{2, 2},
			},
		},
		{
			name: "ones",
			factory: func() (*Tensor, error) {
				return Ones(2, 2, 2), nil
			},
			expected: &Tensor{
				Data:  []float64{1, 1, 1, 1, 1, 1, 1, 1},
				Shape: []int{2, 2, 2},
			},
		},
		{
			name: "oneslike",
			factory: func() (*Tensor, error) {
				return OnesLike(&Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}}), nil
			},
			expected: &Tensor{
				Data:  []float64{1, 1, 1, 1},
				Shape: []int{2, 2},
			},
		},
		{
			name: "all",
			factory: func() (*Tensor, error) {
				return All(3, 2, 2, 2), nil
			},
			expected: &Tensor{
				Data:  []float64{3, 3, 3, 3, 3, 3, 3, 3},
				Shape: []int{2, 2, 2},
			},
		},
		{
			name: "arange",
			factory: func() (*Tensor, error) {
				return Arange(0, 8, 1)
			},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{8},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.factory()
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		arg       []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar to 1d",
			tensor:   &Tensor{Data: []float64{2}, Shape: []int{}},
			arg:      []int{1},
			expected: &Tensor{Data: []float64{2}, Shape: []int{1}},
		},
		{
			name:     "scalar to 2d",
			tensor:   &Tensor{Data: []float64{2}, Shape: []int{}},
			arg:      []int{1, 1},
			expected: &Tensor{Data: []float64{2}, Shape: []int{1, 1}},
		},
		{
			name:     "1d to 2d",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			arg:      []int{1, 3},
			expected: &Tensor{Data: []float64{1, 2, 3}, Shape: []int{1, 3}},
		},
		{
			name:     "2d to 2d",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			arg:      []int{4, 2},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{4, 2}},
		},
		{
			name:     "2d to 3d",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			arg:      []int{1, 2, 4},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{1, 2, 4}},
		},
		{
			name:     "3d to 3d",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 2, 2}},
			arg:      []int{1, 2, 4},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{1, 2, 4}},
		},
		{
			name:     "3d to 2d",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 2, 2}},
			arg:      []int{2, 4},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{2, 4}},
		},
		{
			name:     "3d to 4d",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 2, 2}},
			arg:      []int{1, 1, 2, 4},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{1, 1, 2, 4}},
		},
		{
			name:      "error",
			tensor:    &Tensor{Data: seq(0, 8), Shape: []int{2, 2, 2}},
			arg:       []int{4},
			expectErr: true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Reshape(tc.arg...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestCopy(t *testing.T) {
	t1 := &Tensor{Data: seq(0, 8), Shape: []int{8}}
	t2 := t1.Copy()
	t2, _ = t2.Reshape(2, 2, 2)

	mustEq(t, &Tensor{Data: seq(0, 8), Shape: []int{8}}, t1)
	mustEq(t, &Tensor{Data: seq(0, 8), Shape: []int{2, 2, 2}}, t2)
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		args      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			tensor:   &Tensor{Data: []float64{1}, Shape: []int{}},
			expected: &Tensor{Data: []float64{1}, Shape: []int{}},
		},
		{
			name:     "vector1",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{8}},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{8}},
		},
		{
			name:     "vector2",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{8}},
			args:     []int{0},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{8}},
		},
		{
			name:      "vector err",
			tensor:    &Tensor{Data: seq(0, 8), Shape: []int{8}},
			args:      []int{1},
			expectErr: true,
		},
		{
			name:     "2d1",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			expected: &Tensor{Data: []float64{0, 4, 1, 5, 2, 6, 3, 7}, Shape: []int{4, 2}},
		},
		{
			name:     "2d2",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			args:     []int{1, 0},
			expected: &Tensor{Data: []float64{0, 4, 1, 5, 2, 6, 3, 7}, Shape: []int{4, 2}},
		},
		{
			name:     "2d3 no change",
			tensor:   &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			args:     []int{0, 1},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7}, Shape: []int{2, 4}},
		},
		{
			name:      "2d4 error",
			tensor:    &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			args:      []int{0, 0},
			expectErr: true,
		},
		{
			name:      "2d5 error",
			tensor:    &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			args:      []int{0, 1, 2},
			expectErr: true,
		},
		{
			name:      "2d6 error",
			tensor:    &Tensor{Data: seq(0, 8), Shape: []int{2, 4}},
			args:      []int{0},
			expectErr: true,
		},
		{
			name:     "3d1",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			expected: &Tensor{Data: []float64{0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}, Shape: []int{4, 2, 2}},
		},
		{
			name:     "3d2",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{2, 1, 0},
			expected: &Tensor{Data: []float64{0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}, Shape: []int{4, 2, 2}},
		},
		{
			name:     "3d3",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{0, 1, 2},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, Shape: []int{2, 2, 4}},
		},
		{
			name:     "3d4",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{1, 0, 2},
			expected: &Tensor{Data: []float64{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15}, Shape: []int{2, 2, 4}},
		},
		{
			name:     "3d5",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{2, 0, 1},
			expected: &Tensor{Data: []float64{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, Shape: []int{4, 2, 2}},
		},
		{
			name:     "3d6",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{1, 2, 0},
			expected: &Tensor{Data: []float64{0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15}, Shape: []int{2, 4, 2}},
		},
		{
			name:     "3d7",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{2, 2, 4}},
			args:     []int{0, 2, 1},
			expected: &Tensor{Data: []float64{0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15}, Shape: []int{2, 4, 2}},
		},
		{
			name:     "4d1",
			tensor:   &Tensor{Data: seq(0, 16), Shape: []int{1, 2, 2, 4}},
			expected: &Tensor{Data: []float64{0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15}, Shape: []int{4, 2, 2, 1}},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Transpose(tc.args...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestValueIndices(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		expected []*ValueIndex
	}{
		{
			name:     "scalar",
			tensor:   &Tensor{Data: []float64{1}, Shape: []int{}},
			expected: []*ValueIndex{{[]int{}, 1}},
		},
		{
			name:     "vector",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			expected: []*ValueIndex{{[]int{0}, 1}, {[]int{1}, 2}, {[]int{2}, 3}},
		},
		{
			name:   "2d",
			tensor: &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			expected: []*ValueIndex{
				{[]int{0, 0}, 1}, {[]int{0, 1}, 2},
				{[]int{1, 0}, 3}, {[]int{1, 1}, 4},
				{[]int{2, 0}, 5}, {[]int{2, 1}, 6},
				{[]int{3, 0}, 7}, {[]int{3, 1}, 8},
			},
		},
		{
			name:   "3d",
			tensor: &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{2, 2, 2}},
			expected: []*ValueIndex{
				{[]int{0, 0, 0}, 1}, {[]int{0, 0, 1}, 2},
				{[]int{0, 1, 0}, 3}, {[]int{0, 1, 1}, 4},
				{[]int{1, 0, 0}, 5}, {[]int{1, 0, 1}, 6},
				{[]int{1, 1, 0}, 7}, {[]int{1, 1, 1}, 8},
			},
		},
		{
			name:   "4d",
			tensor: &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, Shape: []int{2, 2, 2, 2}},
			expected: []*ValueIndex{
				{[]int{0, 0, 0, 0}, 1}, {[]int{0, 0, 0, 1}, 2},
				{[]int{0, 0, 1, 0}, 3}, {[]int{0, 0, 1, 1}, 4},
				{[]int{0, 1, 0, 0}, 5}, {[]int{0, 1, 0, 1}, 6},
				{[]int{0, 1, 1, 0}, 7}, {[]int{0, 1, 1, 1}, 8},
				{[]int{1, 0, 0, 0}, 9}, {[]int{1, 0, 0, 1}, 10},
				{[]int{1, 0, 1, 0}, 11}, {[]int{1, 0, 1, 1}, 12},
				{[]int{1, 1, 0, 0}, 13}, {[]int{1, 1, 0, 1}, 14},
				{[]int{1, 1, 1, 0}, 15}, {[]int{1, 1, 1, 1}, 16},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.tensor.ValueIndices()

			if len(got) != len(tc.expected) {
				t.Errorf("expected %v but got %v", tc.expected, got)
			}

			for i := range got {
				g := got[i]
				e := tc.expected[i]
				if !slices.Equal(g.Idx, e.Idx) || g.Value != e.Value {
					t.Errorf("expected %v but got %v", e, g)
				}
			}
		})
	}
}

func TestSubTensor(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		args      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:     "scalar",
			tensor:   &Tensor{Data: []float64{1}, Shape: []int{}},
			args:     nil,
			expected: &Tensor{Data: []float64{1}, Shape: []int{}},
		},
		{
			name:      "scalar err",
			tensor:    &Tensor{Data: []float64{1}, Shape: []int{}},
			args:      []int{0},
			expectErr: true,
		},
		{
			name:     "vector",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:     []int{},
			expected: &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
		},
		{
			name:     "vector 1",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:     []int{0},
			expected: &Tensor{Data: []float64{1}, Shape: []int{}},
		},
		{
			name:     "vector 2",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:     []int{1},
			expected: &Tensor{Data: []float64{2}, Shape: []int{}},
		},
		{
			name:     "vector 3",
			tensor:   &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:     []int{2},
			expected: &Tensor{Data: []float64{3}, Shape: []int{}},
		},
		{
			name:      "vector err",
			tensor:    &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:      []int{3},
			expectErr: true,
		},
		{
			name:      "vector err 2",
			tensor:    &Tensor{Data: []float64{1, 2, 3}, Shape: []int{3}},
			args:      []int{0, 1},
			expectErr: true,
		},
		{
			name:     "2d 1",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:     []int{},
			expected: &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
		},
		{
			name:     "2d 2",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:     []int{0},
			expected: &Tensor{Data: []float64{1, 2}, Shape: []int{2}},
		},
		{
			name:     "2d 3",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:     []int{3},
			expected: &Tensor{Data: []float64{7, 8}, Shape: []int{2}},
		},
		{
			name:     "2d 4",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:     []int{1, 1},
			expected: &Tensor{Data: []float64{4}, Shape: []int{}},
		},
		{
			name:     "2d 5",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:     []int{1, 1},
			expected: &Tensor{Data: []float64{4}, Shape: []int{}},
		},
		{
			name:      "2d err 1",
			tensor:    &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:      []int{4},
			expectErr: true,
		},
		{
			name:      "2d err 2",
			tensor:    &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:      []int{1, 2},
			expectErr: true,
		},
		{
			name:      "2d err 3",
			tensor:    &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{4, 2}},
			args:      []int{-1},
			expectErr: true,
		},
		{
			name:     "3d 1",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{2, 2, 2}},
			args:     []int{0},
			expected: &Tensor{Data: []float64{1, 2, 3, 4}, Shape: []int{2, 2}},
		},
		{
			name:     "3d 2",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{2, 2, 2}},
			args:     []int{1, 1},
			expected: &Tensor{Data: []float64{7, 8}, Shape: []int{2}},
		},
		{
			name:     "3d 3",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{2, 2, 2}},
			args:     []int{1, 1, 1},
			expected: &Tensor{Data: []float64{8}, Shape: []int{}},
		},
		{
			name:     "4d 1",
			tensor:   &Tensor{Data: []float64{1, 2, 3, 4, 5, 6, 7, 8}, Shape: []int{1, 2, 2, 2}},
			args:     []int{0, 1, 0},
			expected: &Tensor{Data: []float64{5, 6}, Shape: []int{2}},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.SubTensor(tc.args)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestTile(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		times     []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "scalar",
			data:  []float64{1},
			shape: []int{},
			times: []int{2},
			expected: &Tensor{
				Data:  []float64{1, 1},
				Shape: []int{2},
			},
		},
		{
			name:  "vector",
			data:  seq(0, 4),
			shape: []int{4},
			times: []int{2, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
				Shape: []int{2, 8},
			},
		},
		{
			name:  "2d",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: []int{2, 1},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 0, 1, 2, 3},
				Shape: []int{4, 2},
			},
		},
		{
			name:  "2d 2",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: []int{2, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3},
				Shape: []int{4, 4},
			},
		},
		{
			name:  "3d 1",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			times: []int{2, 1, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7},
				Shape: []int{4, 2, 4},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Tile(tc.times...)
			if (err != nil) != tc.expectErr {
				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestBroadcastTo(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		btshape   []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:    "scalar",
			data:    []float64{2},
			shape:   []int{},
			btshape: []int{1},
			expected: &Tensor{
				Data:  []float64{2},
				Shape: []int{1},
			},
		},
		{
			name:    "scalar2",
			data:    []float64{2},
			shape:   []int{},
			btshape: []int{1, 1},
			expected: &Tensor{
				Data:  []float64{2},
				Shape: []int{1, 1},
			},
		},
		{
			name:    "vector",
			data:    seq(0, 4),
			shape:   []int{4},
			btshape: []int{2, 4},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 0, 1, 2, 3},
				Shape: []int{2, 4},
			},
		},
		{
			name:    "2d",
			data:    seq(0, 4),
			shape:   []int{2, 2},
			btshape: []int{2, 2, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 0, 1, 2, 3},
				Shape: []int{2, 2, 2},
			},
		},
		{
			name:    "3d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			btshape: []int{4, 2, 2, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{4, 2, 2, 2},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.BroadcastTo(tc.btshape...)
			if (err != nil) != tc.expectErr {
				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestSum(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		keepdims  bool
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "scalar",
			data:  []float64{2},
			shape: []int{},
			expected: &Tensor{
				Data:  []float64{2},
				Shape: []int{},
			},
		},
		{
			name:     "vector 1",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: false,
			expected: &Tensor{
				Data:  []float64{6},
				Shape: []int{},
			},
		},
		{
			name:     "vector 2",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: true,
			expected: &Tensor{
				Data:  []float64{6},
				Shape: []int{1},
			},
		},
		{
			name:     "vector 3",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: true,
			axes:     []int{0},
			expected: &Tensor{
				Data:  []float64{6},
				Shape: []int{1},
			},
		},
		{
			name:  "2d 1",
			data:  seq(0, 6),
			shape: []int{2, 3},
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{},
			},
		},
		{
			name:  "2d 1",
			data:  seq(0, 6),
			shape: []int{2, 3},
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{},
			},
		},
		{
			name:     "2d 2",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			keepdims: true,
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{1, 1},
			},
		},
		{
			name:  "2d 3",
			data:  seq(0, 6),
			shape: []int{2, 3},
			axes:  []int{0},
			expected: &Tensor{
				Data:  []float64{3, 5, 7},
				Shape: []int{3},
			},
		},
		{
			name:  "2d 4",
			data:  seq(0, 6),
			shape: []int{2, 3},
			axes:  []int{1, 0},
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{},
			},
		},
		{
			name:     "2d 5",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			axes:     []int{1},
			keepdims: true,
			expected: &Tensor{
				Data:  []float64{3, 12},
				Shape: []int{2, 1},
			},
		},
		{
			name:     "2d 6",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			axes:     []int{1, 0},
			keepdims: true,
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{1, 1},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Sum(tc.keepdims, tc.axes...)
			if (err != nil) != tc.expectErr {
				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestSqueeze(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "2d 1",
			data:  seq(0, 6),
			shape: []int{2, 3},
			axes:  []int{},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{2, 3},
			},
		},
		{
			name:  "3d 2",
			data:  seq(0, 6),
			shape: []int{1, 2, 3},
			axes:  []int{},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{2, 3},
			},
		},
		{
			name:  "3d 3",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{6},
			},
		},
		{
			name:  "3d 4",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{0},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{6, 1},
			},
		},
		{
			name:  "3d 5",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{1, 6},
			},
		},
		{
			name:  "3d 6",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{0, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5},
				Shape: []int{6},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Squeeze(tc.axes...)
			if (err != nil) != tc.expectErr {
				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestSumTo(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		stshape   []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:    "2d 1",
			data:    seq(0, 6),
			shape:   []int{2, 3},
			stshape: []int{1, 3},
			expected: &Tensor{
				Data:  []float64{3, 5, 7},
				Shape: []int{1, 3},
			},
		},
		{
			name:    "2d 2",
			data:    seq(0, 6),
			shape:   []int{2, 3},
			stshape: []int{2, 1},
			expected: &Tensor{
				Data:  []float64{3, 12},
				Shape: []int{2, 1},
			},
		},
		{
			name:    "2d 3",
			data:    seq(0, 6),
			shape:   []int{2, 3},
			stshape: []int{1, 1},
			expected: &Tensor{
				Data:  []float64{15},
				Shape: []int{1, 1},
			},
		},
		{
			name:    "3d 1",
			data:    seq(0, 24),
			shape:   []int{2, 3, 4},
			stshape: []int{2, 1, 1},
			expected: &Tensor{
				Data:  []float64{66, 210},
				Shape: []int{2, 1, 1},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.SumTo(tc.stshape...)
			if (err != nil) != tc.expectErr {
				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}
