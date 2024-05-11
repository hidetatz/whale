package tensor2

import "testing"

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
			tensor:      Vector([]float64{0, 1, 5, 3, 2, 4}),
			keepdims:    false,
			axis:        -1,
			expectedMax: Scalar(2),
			expectedMin: Scalar(0),
		},
		{
			name:        "vector 2",
			tensor:      Vector([]float64{0, 1, 5, 3, 2, 4}),
			keepdims:    false,
			axis:        0,
			expectedMax: Scalar(2),
			expectedMin: Scalar(0),
		},
		{
			name:        "vector 3",
			tensor:      Vector([]float64{0, 1, 5, 3, 2, 4}),
			keepdims:    true,
			axis:        0,
			expectedMax: Vector([]float64{2}),
			expectedMin: Vector([]float64{0}),
		},
		{
			name:      "vector err",
			tensor:    Vector([]float64{0, 1, 5, 3, 2, 4}),
			keepdims:  true,
			axis:      1,
			expectErr: true,
		},
		{
			name: "3d",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    false,
			axis:        -1,
			expectedMax: Scalar(6),
			expectedMin: Scalar(7),
		},
		{
			name: "3d 2",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    true,
			axis:        -1,
			expectedMax: Must(NdShape([]float64{6}, 1, 1, 1)),
			expectedMin: Must(NdShape([]float64{7}, 1, 1, 1)),
		},
		{
			name: "3d 3",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    false,
			axis:        0,
			expectedMax: Must(NdShape([]float64{1, 0, 1, 1, 0, 0}, 3, 2)),
			expectedMin: Must(NdShape([]float64{0, 1, 0, 0, 1, 1}, 3, 2)),
		},
		{
			name: "3d 4",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    true,
			axis:        0,
			expectedMax: Must(NdShape([]float64{1, 0, 1, 1, 0, 0}, 1, 3, 2)),
			expectedMin: Must(NdShape([]float64{0, 1, 0, 0, 1, 1}, 1, 3, 2)),
		},
		{
			name: "3d 5",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    false,
			axis:        1,
			expectedMax: Must(NdShape([]float64{2, 2, 0, 1}, 2, 2)),
			expectedMin: Must(NdShape([]float64{0, 0, 2, 0}, 2, 2)),
		},
		{
			name: "3d 6",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    true,
			axis:        1,
			expectedMax: Must(NdShape([]float64{2, 2, 0, 1}, 2, 1, 2)),
			expectedMin: Must(NdShape([]float64{0, 0, 2, 0}, 2, 1, 2)),
		},
		{
			name: "3d 7",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    false,
			axis:        2,
			expectedMax: Must(NdShape([]float64{1, 1, 1, 0, 0, 1}, 2, 3)),
			expectedMin: Must(NdShape([]float64{0, 0, 0, 1, 1, 0}, 2, 3)),
		},
		{
			name: "3d 8",
			tensor: Must(New([][][]float64{
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
			})),
			keepdims:    true,
			axis:        2,
			expectedMax: Must(NdShape([]float64{1, 1, 1, 0, 0, 1}, 2, 3, 1)),
			expectedMin: Must(NdShape([]float64{0, 0, 0, 1, 1, 0}, 2, 3, 1)),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			max, err := tc.tensor.Argmax(tc.keepdims, tc.axis)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expectedMax, max)

			min, err := tc.tensor.Argmin(tc.keepdims, tc.axis)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expectedMin, min)
		})
	}
}
