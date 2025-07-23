package tensor

import "testing"

func TestMinMax(t *testing.T) {
	tests := []struct {
		name        string
		tensor      *Tensor
		expectedMin *Tensor
		expectedMax *Tensor
	}{
		{
			name:        "scalar",
			tensor:      Scalar(3),
			expectedMin: Scalar(3),
			expectedMax: Scalar(3),
		},
		{
			name:        "vector",
			tensor:      Vector([]float32{0, 1, 2, 3, 4}),
			expectedMin: Scalar(0),
			expectedMax: Scalar(4),
		},
		{
			name:        "2d",
			tensor:      NdShape([]float32{0, 1, 2, 3}, 2, 2),
			expectedMin: Scalar(0),
			expectedMax: Scalar(3),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			min := tc.tensor.Min()
			mustEq(t, tc.expectedMin, min)
			max := tc.tensor.Max()
			mustEq(t, tc.expectedMax, max)
		})
	}
}
