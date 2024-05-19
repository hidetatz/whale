package tensor

import "testing"

func TestMeshGrid(t *testing.T) {
	tests := []struct {
		name      string
		t1, t2    *Tensor
		ex1, ex2  *Tensor
		expectErr bool
	}{
		{
			name:      "invalid shape",
			t1:        Scalar(3),
			t2:        Vector([]float32{1, 2, 3, 4}),
			expectErr: true,
		},
		{
			name: "valid",
			t1:   Vector([]float32{0, 1, 2, 3}),
			t2:   Vector([]float32{10, 20, 30, 40, 50, 60}),
			ex1: New([][]float32{
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
				{0, 1, 2, 3},
			}),
			ex2: New([][]float32{
				{10, 10, 10, 10},
				{20, 20, 20, 20},
				{30, 30, 30, 30},
				{40, 40, 40, 40},
				{50, 50, 50, 50},
				{60, 60, 60, 60},
			}),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got1, got2, err := RespErr.MeshGrid(tc.t1, tc.t2)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.ex1, got1)
			mustEq(t, tc.ex2, got2)
		})
	}
}
