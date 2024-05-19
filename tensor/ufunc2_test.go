package tensor

import "testing"

func TestUniversalfunc2_Do(t *testing.T) {
	tests := []struct {
		name      string
		x         *Tensor
		target    *Tensor
		ufunc     *universalfunc2
		expected  *Tensor
		expectErr bool
	}{
		{
			name:     "scalar",
			x:        Scalar(3),
			target:   Scalar(1),
			ufunc:    ADD,
			expected: Scalar(4),
		},
		{
			name:     "vector",
			x:        Vector([]float32{1, 2, 3, 4, 5}),
			target:   Scalar(1),
			ufunc:    ADD,
			expected: Vector([]float32{2, 3, 4, 5, 6}),
		},
		{
			name:     "2d",
			x:        Vector([]float32{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			target:   Vector([]float32{1, 2, 3}),
			ufunc:    MUL,
			expected: Vector([]float32{1, 4, 9, 4, 10, 18}).Reshape(2, 3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.ufunc.Do(tc.x, tc.target)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestUniversalfunc2_At(t *testing.T) {
	tests := []struct {
		name      string
		x         *Tensor
		indices   []*IndexArg
		target    *Tensor
		expected  *Tensor
		expectErr bool
	}{
		{
			name:     "error empty",
			x:        Arange(3, 6, 1),
			indices:  []*IndexArg{At(0)},
			target:   Scalar(1),
			expected: Vector([]float32{4, 4, 5}),
		},
		{
			name:      "scalar error",
			x:         Scalar(3),
			indices:   []*IndexArg{At(0)},
			target:    Scalar(1),
			expectErr: true,
			expected:  Scalar(3),
		},
		{
			name:     "vector",
			x:        Vector([]float32{1, 2, 3, 4, 5}),
			indices:  []*IndexArg{At(0)},
			target:   Scalar(1),
			expected: Vector([]float32{2, 2, 3, 4, 5}),
		},
		{
			name: "vector 2",
			x:    Vector([]float32{1, 2, 3, 4, 5}),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1, 2})),
			},
			target:   Scalar(1),
			expected: Vector([]float32{2, 3, 4, 4, 5}),
		},
		{
			name: "vector 3",
			x:    Vector([]float32{1, 2, 3, 4, 5}),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1, 2})),
			},
			target:   Vector([]float32{1, 2, 3}),
			expected: Vector([]float32{2, 4, 6, 4, 5}),
		},
		{
			name: "vector err",
			x:    Vector([]float32{1, 2, 3, 4, 5}),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1, 2})),
				List(Vector([]float32{0, 1, 2})),
			},
			target:    Scalar(1),
			expectErr: true,
			expected:  Vector([]float32{1, 2, 3, 4, 5}),
		},
		{
			name: "vector err2",
			x:    Vector([]float32{1, 2, 3, 4, 5}),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1, 2})),
			},
			target:    Vector([]float32{1, 2}),
			expectErr: true,
			expected:  Vector([]float32{1, 2, 3, 4, 5}),
		},
		{
			name: "2d",
			x:    Arange(1, 7, 1).Reshape(2, 3),
			indices: []*IndexArg{
				List(Vector([]float32{0, 0, 1})),
			},
			target:   Scalar(1),
			expected: NdShape([]float32{3, 4, 5, 5, 6, 7}, 2, 3),
		},
		{
			name: "2d 2",
			x:    Arange(1, 7, 1).Reshape(2, 3),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1})),
				List(Vector([]float32{1, 2})),
			},
			target:   Scalar(1),
			expected: NdShape([]float32{1, 3, 3, 4, 5, 7}, 2, 3),
		},
		{
			name: "2d 3",
			x:    Arange(1, 7, 1).Reshape(2, 3),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1})),
				List(Vector([]float32{1, 2})),
			},
			target:   Vector([]float32{1, 2}),
			expected: NdShape([]float32{1, 3, 3, 4, 5, 8}, 2, 3),
		},
		{
			name: "2d err",
			x:    Arange(1, 7, 1).Reshape(2, 3),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1})),
				List(Vector([]float32{1, 2})),
				List(Vector([]float32{0, 0})),
			},
			target:    Scalar(1),
			expectErr: true,
			expected:  Arange(1, 7, 1).Reshape(2, 3),
		},
		{
			name: "2d err 2",
			x:    Arange(1, 7, 1).Reshape(2, 3),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1})),
				List(Vector([]float32{1, 2})),
			},
			target:    Vector([]float32{1, 2, 3}),
			expectErr: true,
			expected:  Arange(1, 7, 1).Reshape(2, 3),
		},
		{
			name: "3d",
			x:    Arange(1, 13, 1).Reshape(2, 3, 2),
			indices: []*IndexArg{
				List(Vector([]float32{0, 1})),
				List(Vector([]float32{1, 2})),
				List(Vector([]float32{1, 0})),
			},
			target:   Vector([]float32{1, 2}),
			expected: NdShape([]float32{1, 2, 3, 5, 5, 6, 7, 8, 9, 10, 13, 12}, 2, 3, 2),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := ADD.At(tc.x, tc.indices, tc.target)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, tc.x)
		})
	}
}
