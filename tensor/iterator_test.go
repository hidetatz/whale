package tensor

import (
	"slices"
	"testing"
)

func TestIterator(t *testing.T) {
	tests := []struct {
		name           string
		tensor         *Tensor
		expectedIndex  []int
		expectedValues []float32
	}{
		{
			name:           "scalar",
			tensor:         Scalar(3),
			expectedIndex:  []int{0},
			expectedValues: []float32{3},
		},
		{
			name:           "vector",
			tensor:         Vector([]float32{5, 4, 3, 2, 1}),
			expectedIndex:  []int{0, 1, 2, 3, 4},
			expectedValues: []float32{5, 4, 3, 2, 1},
		},
		{
			name: "2d",
			tensor: New([][]float32{
				{1, 2},
				{3, 4},
			}),
			expectedIndex:  []int{0, 1, 2, 3},
			expectedValues: []float32{1, 2, 3, 4},
		},
		{
			name: "3d",
			tensor: New([][][]float32{
				{
					{1, 2},
					{3, 4},
				},
				{
					{5, 6},
					{7, 8},
				},
			}),
			expectedIndex:  []int{0, 1, 2, 3, 4, 5, 6, 7},
			expectedValues: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			iter := tc.tensor.Iterator()
			indices := []int{}
			values := []float32{}
			for iter.HasNext() {
				i, v := iter.Next()
				indices = append(indices, i)
				values = append(values, v)
			}

			if !slices.Equal(indices, tc.expectedIndex) {
				t.Fatalf("index wrong: expected: %v, got: %v", tc.expectedIndex, indices)
			}

			if !slices.Equal(values, tc.expectedValues) {
				t.Fatalf("values wrong: expected: %v, got: %v", tc.expectedValues, values)
			}
		})
	}
}
