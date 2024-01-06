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
			got := FromScalar(tc.arg)
			if !tc.expected.Equals(got) {
				t.Errorf("expected %v but got %v", tc.expected, got)
			}
		})
	}
}

func TestFromVector(t *testing.T) {
	tests := []struct {
		name        string
		data        []float64
		shape       int
		expectError bool
		expected    *Tensor
	}{
		{
			name:  "simple",
			data:  []float64{1, 2, 3},
			shape: 3,
			expected: &Tensor{
				Data:    []float64{1, 2, 3},
				shape:   []int{3},
				strides: []int{1},
			},
		},
		{
			name:  "empty",
			data:  []float64{},
			shape: 0,
			expected: &Tensor{
				Data:    []float64{},
				shape:   []int{0},
				strides: []int{1},
			},
		},
		{
			name:        "shape mismatch",
			data:        []float64{1, 2, 3},
			shape:       2,
			expectError: true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := FromVector(tc.data, tc.shape)
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

func TestNd(t *testing.T) {
	tests := []struct {
		name        string
		data        []float64
		shape       []int
		expectError bool
		expected    *Tensor
	}{
		{
			name:  "scalar",
			data:  []float64{1},
			shape: []int{},
			expected: &Tensor{
				Data:    []float64{1},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:  "vector",
			data:  []float64{1, 2},
			shape: []int{2},
			expected: &Tensor{
				Data:    []float64{1, 2},
				shape:   []int{2},
				strides: []int{1},
			},
		},
		{
			name:  "matrix",
			data:  []float64{1, 2, 3, 4},
			shape: []int{2, 2},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{2, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:  "matrix2",
			data:  []float64{1, 2, 3, 4},
			shape: []int{1, 4},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{1, 4},
				strides: []int{4, 1},
			},
		},
		{
			name:  "matrix3",
			data:  []float64{1, 2, 3, 4},
			shape: []int{4, 1},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{4, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:  "3d",
			data:  []float64{1, 2, 3, 4},
			shape: []int{4, 1, 1},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{4, 1, 1},
				strides: []int{1, 1, 1},
			},
		},
		{
			name:  "3d2",
			data:  []float64{1, 2, 3, 4},
			shape: []int{2, 2, 1},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{2, 2, 1},
				strides: []int{2, 1, 1},
			},
		},
		{
			name:  "4d",
			data:  []float64{1, 2, 3, 4},
			shape: []int{1, 1, 1, 4},
			expected: &Tensor{
				Data:    []float64{1, 2, 3, 4},
				shape:   []int{1, 1, 1, 4},
				strides: []int{4, 4, 4, 1},
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
				Data:    []float64{0, 0, 0, 0, 0, 0, 0, 0},
				shape:   []int{2, 2, 2},
				strides: []int{4, 2, 1},
			},
		},
		{
			name: "ones",
			factory: func() (*Tensor, error) {
				return Ones(2, 2, 2), nil
			},
			expected: &Tensor{
				Data:    []float64{1, 1, 1, 1, 1, 1, 1, 1},
				shape:   []int{2, 2, 2},
				strides: []int{4, 2, 1},
			},
		},
		{
			name: "arange",
			factory: func() (*Tensor, error) {
				return ArangeTo(8), nil
			},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{8},
				strides: []int{1},
			},
		},
		{
			name: "arange2",
			factory: func() (*Tensor, error) {
				return ArangeFrom(4, 12), nil
			},
			expected: &Tensor{
				Data:    []float64{4, 5, 6, 7, 8, 9, 10, 11},
				shape:   []int{8},
				strides: []int{1},
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
				Data:    []float64{2},
				shape:   []int{1},
				strides: []int{1},
			},
		},
		{
			name:    "scalar2",
			data:    []float64{2},
			shape:   []int{},
			reshape: []int{1, 1},
			expected: &Tensor{
				Data:    []float64{2},
				shape:   []int{1, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:    "vector",
			data:    []float64{1, 2, 3},
			shape:   []int{3},
			reshape: []int{1, 3},
			expected: &Tensor{
				Data:    []float64{1, 2, 3},
				shape:   []int{1, 3},
				strides: []int{3, 1},
			},
		},
		{
			name:    "1d to 2d",
			data:    seq(0, 8),
			shape:   []int{8},
			reshape: []int{4, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{4, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:    "2d to 2d",
			data:    seq(0, 8),
			shape:   []int{2, 4},
			reshape: []int{4, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{4, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:    "2d to 3d",
			data:    seq(0, 8),
			shape:   []int{2, 4},
			reshape: []int{1, 2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{1, 2, 4},
				strides: []int{8, 4, 1},
			},
		},
		{
			name:    "3d to 3d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			reshape: []int{1, 2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{1, 2, 4},
				strides: []int{8, 4, 1},
			},
		},
		{
			name:    "3d to 4d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			reshape: []int{1, 1, 2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{1, 1, 2, 4},
				strides: []int{8, 8, 4, 1},
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
			name:  "scalar",
			data:  []float64{3},
			shape: []int{},
			axes:  []int{},
			expected: &Tensor{
				Data:    []float64{3},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:  "vector",
			data:  seq(0, 8),
			shape: []int{8},
			axes:  []int{0},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{8},
				strides: []int{1},
			},
		},
		{
			name:  "scalar",
			data:  []float64{1},
			shape: nil,
			expected: &Tensor{
				Data:    []float64{1},
				shape:   nil,
				strides: nil,
			},
		},
		{
			name:  "vector",
			data:  seq(0, 8),
			shape: []int{8},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{8},
				strides: []int{1},
			},
		},
		{
			name:  "2d",
			data:  seq(0, 8),
			shape: []int{2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{4, 2},
				strides: []int{1, 4},
			},
		},
		{
			name:  "3d",
			data:  seq(0, 16),
			shape: []int{2, 2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
				shape:   []int{4, 2, 2},
				strides: []int{1, 4, 8},
			},
		},
		{
			name:  "4d",
			data:  seq(0, 16),
			shape: []int{1, 2, 2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
				shape:   []int{4, 2, 2, 1},
				strides: []int{1, 4, 8, 16},
			},
		},
		{
			name:  "2d nochange",
			data:  seq(0, 8),
			shape: []int{2, 4},
			axes:  []int{0, 1},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{2, 4},
				strides: []int{4, 1},
			},
		},
		{
			name:  "2d",
			data:  seq(0, 8),
			shape: []int{2, 4},
			axes:  []int{1, 0},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{4, 2},
				strides: []int{1, 4},
			},
		},
		{
			name:  "3d",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			axes:  []int{1, 0, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{2, 2, 2},
				strides: []int{2, 4, 1},
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

func TestRepeat(t *testing.T) {
	tests := []struct {
		name      string
		data      []float64
		shape     []int
		times     int
		axis      int
		expectErr bool
		expected  *Tensor
	}{
		{
			name:  "vector",
			data:  seq(0, 4),
			shape: []int{4},
			times: 2,
			axis:  0,
			expected: &Tensor{
				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3},
				shape:   []int{8},
				strides: []int{1},
			},
		},
		{
			name:  "2d",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: 2,
			axis:  0,
			expected: &Tensor{
				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3},
				shape:   []int{4, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:  "2d 2",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: 2,
			axis:  1,
			expected: &Tensor{
				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3},
				shape:   []int{2, 4},
				strides: []int{4, 1},
			},
		},
		{
			name:  "3d",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			times: 2,
			axis:  0,
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7},
				shape:   []int{4, 2, 2},
				strides: []int{4, 2, 1},
			},
		},
		{
			name:  "3d 2",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			times: 2,
			axis:  1,
			expected: &Tensor{
				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7},
				shape:   []int{2, 4, 2},
				strides: []int{8, 2, 1},
			},
		},
		{
			name:  "3d 3",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			times: 2,
			axis:  2,
			expected: &Tensor{
				Data:    []float64{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7},
				shape:   []int{2, 2, 4},
				strides: []int{8, 4, 1},
			},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, _ := Nd(tc.data, tc.shape...)
			got, err := got.Repeat(tc.times, tc.axis)
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
				Data:    []float64{1, 1},
				shape:   []int{2},
				strides: []int{1},
			},
		},
		{
			name:  "vector",
			data:  seq(0, 4),
			shape: []int{4},
			times: []int{2, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
				shape:   []int{2, 8},
				strides: []int{8, 1},
			},
		},
		{
			name:  "2d",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: []int{2, 1},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3},
				shape:   []int{4, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:  "2d 2",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: []int{2, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3},
				shape:   []int{4, 4},
				strides: []int{4, 1},
			},
		},
		{
			name:  "2d 3",
			data:  seq(0, 4),
			shape: []int{2, 2},
			times: []int{1, 1},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3},
				shape:   []int{2, 2},
				strides: []int{2, 1},
			},
		},
		{
			name:  "3d 1",
			data:  seq(0, 8),
			shape: []int{2, 2, 2},
			times: []int{2, 1, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7},
				shape:   []int{4, 2, 4},
				strides: []int{8, 4, 1},
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
				Data:    []float64{2},
				shape:   []int{1},
				strides: []int{1},
			},
		},
		{
			name:    "scalar2",
			data:    []float64{2},
			shape:   []int{},
			btshape: []int{1, 1},
			expected: &Tensor{
				Data:    []float64{2},
				shape:   []int{1, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:    "vector",
			data:    seq(0, 4),
			shape:   []int{4},
			btshape: []int{2, 4},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3},
				shape:   []int{2, 4},
				strides: []int{4, 1},
			},
		},
		{
			name:    "2d",
			data:    seq(0, 4),
			shape:   []int{2, 2},
			btshape: []int{2, 2, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 0, 1, 2, 3},
				shape:   []int{2, 2, 2},
				strides: []int{4, 2, 1},
			},
		},
		{
			name:    "3d",
			data:    seq(0, 8),
			shape:   []int{2, 2, 2},
			btshape: []int{4, 2, 2, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
				shape:   []int{4, 2, 2, 2},
				strides: []int{8, 4, 2, 1},
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
				Data:    []float64{2},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:     "vector 1",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: false,
			expected: &Tensor{
				Data:    []float64{6},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:     "vector 2",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: true,
			expected: &Tensor{
				Data:    []float64{6},
				shape:   []int{1},
				strides: []int{1},
			},
		},
		{
			name:     "vector 3",
			data:     seq(0, 4),
			shape:    []int{4},
			keepdims: true,
			axes:     []int{0},
			expected: &Tensor{
				Data:    []float64{6},
				shape:   []int{1},
				strides: []int{1},
			},
		},
		{
			name:  "2d 1",
			data:  seq(0, 6),
			shape: []int{2, 3},
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:  "2d 1",
			data:  seq(0, 6),
			shape: []int{2, 3},
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:     "2d 2",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			keepdims: true,
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{1, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:  "2d 3",
			data:  seq(0, 6),
			shape: []int{2, 3},
			axes:  []int{0},
			expected: &Tensor{
				Data:    []float64{3, 5, 7},
				shape:   []int{3},
				strides: []int{1},
			},
		},
		{
			name:  "2d 4",
			data:  seq(0, 6),
			shape: []int{2, 3},
			axes:  []int{1, 0},
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{},
				strides: []int{},
			},
		},
		{
			name:     "2d 5",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			axes:     []int{1},
			keepdims: true,
			expected: &Tensor{
				Data:    []float64{3, 12},
				shape:   []int{2, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:     "2d 6",
			data:     seq(0, 6),
			shape:    []int{2, 3},
			axes:     []int{1, 0},
			keepdims: true,
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{1, 1},
				strides: []int{1, 1},
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
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{2, 3},
				strides: []int{3, 1},
			},
		},
		{
			name:  "3d 2",
			data:  seq(0, 6),
			shape: []int{1, 2, 3},
			axes:  []int{},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{2, 3},
				strides: []int{3, 1},
			},
		},
		{
			name:  "3d 3",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{6},
				strides: []int{1},
			},
		},
		{
			name:  "3d 4",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{0},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{6, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:  "3d 5",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{1, 6},
				strides: []int{6, 1},
			},
		},
		{
			name:  "3d 6",
			data:  seq(0, 6),
			shape: []int{1, 6, 1},
			axes:  []int{0, 2},
			expected: &Tensor{
				Data:    []float64{0, 1, 2, 3, 4, 5},
				shape:   []int{6},
				strides: []int{1},
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
				Data:    []float64{3, 5, 7},
				shape:   []int{1, 3},
				strides: []int{3, 1},
			},
		},
		{
			name:    "2d 2",
			data:    seq(0, 6),
			shape:   []int{2, 3},
			stshape: []int{2, 1},
			expected: &Tensor{
				Data:    []float64{3, 12},
				shape:   []int{2, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:    "2d 3",
			data:    seq(0, 6),
			shape:   []int{2, 3},
			stshape: []int{1, 1},
			expected: &Tensor{
				Data:    []float64{15},
				shape:   []int{1, 1},
				strides: []int{1, 1},
			},
		},
		{
			name:    "3d 1",
			data:    seq(0, 24),
			shape:   []int{2, 3, 4},
			stshape: []int{2, 1, 1},
			expected: &Tensor{
				Data:    []float64{66, 210},
				shape:   []int{2, 1, 1},
				strides: []int{1, 1, 1},
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
