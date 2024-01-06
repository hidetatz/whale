package tensor

import (
	"testing"
)

func mustEq(t *testing.T, expected, got *Tensor) {
	if !expected.Equals(got) {
		t.Fatalf("expected %v but got %v", expected, got)
	}
}

func TestScalar(t *testing.T) {
	tests := []struct {
		name     string
		arg      float64
		expected *Tensor
	}{
		{
			name:     "simple",
			arg:      0.1,
			expected: &Tensor{Data: []float64{0.1}},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := Scalar(tc.arg)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestVector(t *testing.T) {
	tests := []struct {
		name     string
		data     []float64
		shape    int
		expected *Tensor
	}{
		{
			name:  "simple",
			data:  []float64{1, 2, 3},
			shape: 3,
			expected: &Tensor{
				Data:  []float64{1, 2, 3},
				Shape: []int{3},
			},
		},
		{
			name:  "empty",
			data:  []float64{},
			shape: 0,
			expected: &Tensor{
				Data:  []float64{},
				Shape: []int{0},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := Vector(tc.data)
			if !tc.expected.Equals(got) {
				t.Errorf("expected %v but got %v", tc.expected, got)
			}
		})
	}
}

func TestNd(t *testing.T) {
	tests := []struct {
		name        string
		data        []float64
		shape       []int
		expectError bool
		expected    *Tensor
	}{
		{
			name:  "vector",
			data:  []float64{1, 2},
			shape: []int{2},
			expected: &Tensor{
				Data:  []float64{1, 2},
				Shape: []int{2},
			},
		},
		{
			name:  "matrix",
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
			if (err != nil) != tc.expectError {
				t.Errorf("unexpected error: expected: %v but got %v", tc.expectError, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestFactories(t *testing.T) {
	tests := []struct {
		name        string
		factory     func() (*Tensor, error)
		expectError bool
		expected    *Tensor
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
			if (err != nil) != tc.expectError {
				t.Errorf("unexpected error: expected: %v but got %v", tc.expectError, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestReshape(t *testing.T) {
	tests := []struct {
		name        string
		data        []float64
		shape       []int
		reshape     []int
		expectError bool
		expected    *Tensor
	}{
		{
			name:    "scalar",
			data:    []float64{2},
			shape:   []int{},
			reshape: []int{1},
			expected: &Tensor{
				Data:  []float64{2},
				Shape: []int{1},
			},
		},
		{
			name:    "scalar2",
			data:    []float64{2},
			shape:   []int{},
			reshape: []int{1, 1},
			expected: &Tensor{
				Data:  []float64{2},
				Shape: []int{1, 1},
			},
		},
		{
			name:    "vector",
			data:    []float64{1, 2, 3},
			shape:   []int{3},
			reshape: []int{1, 3},
			expected: &Tensor{
				Data:  []float64{1, 2, 3},
				Shape: []int{1, 3},
			},
		},
		{
			name:    "1d to 2d",
			data:    seq(0, 8),
			shape:   []int{8},
			reshape: []int{4, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{4, 2},
			},
		},
		{
			name:    "2d to 2d",
			data:    seq(0, 8),
			shape:   []int{2, 4},
			reshape: []int{4, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{4, 2},
			},
		},
		{
			name:    "2d to 3d",
			data:    seq(0, 8),
			shape:   []int{2, 4},
			reshape: []int{1, 2, 4},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{1, 2, 4},
			},
		},
		{
			name:    "3d to 3d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			reshape: []int{1, 2, 4},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{1, 2, 4},
			},
		},
		{
			name:    "3d to 4d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			reshape: []int{1, 1, 2, 4},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{1, 1, 2, 4},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Reshape(tc.reshape...)
			if (err != nil) != tc.expectError {
				t.Errorf("unexpected error: expected: %v but got %v", tc.expectError, err)
			}

			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		axes      []int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "vector",
			data:  seq(0, 8),
			shape: []int{8},
			axes:  []int{0},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{8},
			},
		},
		{
			name:  "vector",
			data:  seq(0, 8),
			shape: []int{8},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{8},
			},
		},
		{
			name:  "2d1",
			data:  seq(0, 8),
			shape: []int{2, 4},
			expected: &Tensor{
				Data:  []float64{0, 4, 1, 5, 2, 6, 3, 7},
				Shape: []int{4, 2},
			},
		},
		{
			name:  "2d nochange",
			data:  seq(0, 8),
			shape: []int{2, 4},
			axes:  []int{0, 1},
			expected: &Tensor{
				Data:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
				Shape: []int{2, 4},
			},
		},
		{
			name:  "2d2",
			data:  seq(0, 8),
			shape: []int{2, 4},
			axes:  []int{1, 0},
			expected: &Tensor{
				Data:  []float64{0, 4, 1, 5, 2, 6, 3, 7},
				Shape: []int{4, 2},
			},
		},
		{
			name:  "3d1",
			data:  seq(0, 16),
			shape: []int{2, 2, 4},
			expected: &Tensor{
				Data:  []float64{0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15},
				Shape: []int{4, 2, 2},
			},
		},
		{
			name:  "3d2",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			axes:  []int{1, 0, 2},
			expected: &Tensor{
				Data:  []float64{0, 1, 4, 5, 2, 3, 6, 7},
				Shape: []int{2, 2, 2},
			},
		},
		{
			name:  "4d1",
			data:  seq(0, 16),
			shape: []int{1, 2, 2, 4},
			expected: &Tensor{
				Data:  []float64{0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15},
				Shape: []int{4, 2, 2, 1},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Transpose(tc.axes...)
			if (err != nil) != tc.expectErr {
				t.Errorf("unexpected error: expected: %v but got %v", tc.expectErr, err)
			}
			if tc.expected != nil {
				if !tc.expected.Equals(got) {
					t.Errorf("expected %v but got %v", tc.expected, got)
				}
			}
		})
	}
}

// func TestRepeat(t *testing.T) {
// 	tests := []struct {
// 		name      string
// 		data      []float64
// 		shape     []int
// 		times     int
// 		axis      int
// 		expectErr bool
// 		expected  *Tensor
// 	}{
// 		{
// 			name:  "vector",
// 			data:  seq(0, 4),
// 			shape: []int{4},
// 			times: 2,
// 			axis:  0,
// 			expected: &Tensor{
// 				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3},
// 				Shape:   []int{8},
// 			},
// 		},
// 		{
// 			name:  "2d",
// 			data:  seq(0, 4),
// 			shape: []int{2, 2},
// 			times: 2,
// 			axis:  0,
// 			expected: &Tensor{
// 				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3},
// 				Shape:   []int{4, 2},
// 			},
// 		},
// 		{
// 			name:  "2d 2",
// 			data:  seq(0, 4),
// 			shape: []int{2, 2},
// 			times: 2,
// 			axis:  1,
// 			expected: &Tensor{
// 				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3},
// 				Shape:   []int{2, 4},
// 			},
// 		},
// 		{
// 			name:  "3d",
// 			data:  seq(0, 8),
// 			shape: []int{2, 2, 2},
// 			times: 2,
// 			axis:  0,
// 			expected: &Tensor{
// 				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7},
// 				Shape:   []int{4, 2, 2},
// 			},
// 		},
// 		{
// 			name:  "3d 2",
// 			data:  seq(0, 8),
// 			shape: []int{2, 2, 2},
// 			times: 2,
// 			axis:  1,
// 			expected: &Tensor{
// 				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7},
// 				Shape:   []int{2, 4, 2},
// 			},
// 		},
// 		{
// 			name:  "3d 3",
// 			data:  seq(0, 8),
// 			shape: []int{2, 2, 2},
// 			times: 2,
// 			axis:  2,
// 			expected: &Tensor{
// 				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7},
// 				Shape:   []int{2, 2, 4},
// 			},
// 		},
// 	}
//
// 	for _, tc := range tests {
// 		tc := tc
// 		t.Run(tc.name, func(t *testing.T) {
// 			t.Parallel()
// 			got, _ := Nd(tc.data, tc.shape...)
// 			got, err := got.Repeat(tc.times, tc.axis)
// 			if (err != nil) != tc.expectErr {
// 				t.Fatalf("unexpected error: expected: %v but got %v", tc.expectErr, err)
// 			}
// 			if tc.expected != nil {
// 				if !tc.expected.Equals(got) {
// 					t.Errorf("expected %v but got %v", tc.expected, got)
// 				}
// 			}
// 		})
// 	}
// }

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
