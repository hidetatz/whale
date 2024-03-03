package tensor2

// func TestListIndex(t *testing.T) {
// 	tests := []struct {
// 		name      string
// 		tensor    *Tensor
// 		indices   [][]int
// 		expected  *Tensor
// 		expectErr bool
// 	}{
// 		{
// 			name:      "scalar cannot indexed",
// 			tensor:    Scalar(1),
// 			indices:   [][]int{{0}},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "empty index is not allowed",
// 			tensor:    Vector([]float64{1, 2, 3}),
// 			indices:   [][]int{},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "too many indices",
// 			tensor:    Vector([]float64{1, 2, 3}),
// 			indices:   [][]int{{0}, {1}},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "too big index",
// 			tensor:    Vector([]float64{1, 2, 3}),
// 			indices:   [][]int{{3}},
// 			expectErr: true,
// 		},
// 		{
// 			name:     "vector 1",
// 			tensor:   Vector([]float64{1, 2, 3}),
// 			indices:  [][]int{{0}},
// 			expected: Vector([]float64{1}),
// 		},
// 		{
// 			name:     "vector 2",
// 			tensor:   Vector([]float64{1, 2, 3}),
// 			indices:  [][]int{{0, 1, 2}},
// 			expected: Vector([]float64{1, 2, 3}),
// 		},
// 		{
// 			name:     "vector 3",
// 			tensor:   Vector([]float64{1, 2, 3}),
// 			indices:  [][]int{{2, 0, 0, 1, 2, 0}},
// 			expected: Vector([]float64{3, 1, 1, 2, 3, 1}),
// 		},
// 		{
// 			name:     "2d 1",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  [][]int{{0, 1, 0}},
// 			expected: MustNdShape([]float64{1, 2, 3, 4, 5, 6, 1, 2, 3}, 3, 3),
// 		},
// 		{
// 			name:     "2d 2",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  [][]int{{0, 1, 0}, {2, 2, 1}},
// 			expected: Vector([]float64{3, 6, 2}),
// 		},
// 		{
// 			name:    "2d 3",
// 			tensor:  MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices: [][]int{{}},
// 			// empty index itself is not a problem (the same behavior as numpy)
// 			expected: &Tensor{data: []float64{}, Shape: []int{0, 3}, Strides: []int{3, 1}},
// 		},
// 		{
// 			name:     "2d 4",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  [][]int{{}, {}},
// 			expected: Vector([]float64{}),
// 		},
// 		{
// 			name:     "2d 5",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  [][]int{{0, 0, 1}, {2}},
// 			expected: Vector([]float64{3, 3, 6}),
// 		},
// 		{
// 			name:     "2d 3",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  [][]int{{0}, {2}},
// 			expected: Vector([]float64{3}),
// 		},
// 		{
// 			name:      "impossible broadcast",
// 			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:   [][]int{{0, 0, 1}, {2, 1}},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "too many indices",
// 			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:   [][]int{{}, {}, {}},
// 			expectErr: true,
// 		},
// 	}
//
// 	for _, tc := range tests {
// 		tc := tc
// 		t.Run(tc.name, func(t *testing.T) {
// 			t.Parallel()
// 			got, err := tc.tensor.ListIndex(tc.indices)
// 			checkErr(t, tc.expectErr, err)
// 			mustEq(t, tc.expected, got)
// 		})
// 	}
// }
