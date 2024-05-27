package tensor

import "testing"

func TestMatmul(t *testing.T) {
	tests := []struct {
		name      string
		t1, t2    *Tensor
		expected  *Tensor
		expectErr bool
	}{
		{
			name:      "invalid shape 1",
			t1:        Scalar(3),
			t2:        Scalar(4),
			expectErr: true,
		},
		{
			name:      "invalid shape 2",
			t1:        NdShape([]float32{0, 1, 2, 3}, 2, 2),
			t2:        NdShape([]float32{0, 1, 2, 3}, 2, 1, 2),
			expectErr: true,
		},
		{
			name:      "invalid shape 3",
			t1:        NdShape([]float32{0, 1, 2, 3, 4, 5}, 3, 2),
			t2:        NdShape([]float32{0, 1, 2, 3, 4, 5}, 3, 2),
			expectErr: true,
		},
		{
			name: "valid 1",
			t1: New([][]float32{
				{0, 1},
				{2, 3},
			}),
			t2: New([][]float32{
				{0, 1},
				{2, 3},
			}),
			expected: New([][]float32{
				{2, 3},
				{6, 11},
			}),
		},
		{
			name: "valid 2",
			t1: New([][]float32{
				{8, 4, 2},
				{1, 3, -6},
				{-7, 0, 5},
			}),
			t2: New([][]float32{
				{5, 2},
				{3, 1},
				{4, -1},
			}),
			expected: New([][]float32{
				{60, 18},
				{-10, 11},
				{-15, -19},
			}),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.t1.ErrResponser().Matmul(tc.t2)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}
