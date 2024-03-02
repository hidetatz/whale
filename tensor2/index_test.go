package tensor2

import (
	"testing"
)

// func TestIndex(t *testing.T) {
// 	tests := []struct {
// 		name      string
// 		tensor    *Tensor
// 		indices   []int
// 		expected  *Tensor
// 		expectErr bool
// 	}{
// 		{
// 			name:      "scalar cannot indexed",
// 			tensor:    Scalar(1),
// 			indices:   []int{},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "missing index",
// 			tensor:    Vector([]float64{1, 2, 3}),
// 			indices:   []int{},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "too many index",
// 			tensor:    Vector([]float64{1, 2, 3}),
// 			indices:   []int{0, 1},
// 			expectErr: true,
// 		},
// 		{
// 			name:     "vector 1",
// 			tensor:   Vector([]float64{1, 2, 3}),
// 			indices:  []int{0},
// 			expected: Scalar(1),
// 		},
// 		{
// 			name:     "2d 1",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  []int{0},
// 			expected: Vector([]float64{1, 2, 3}),
// 		},
// 		{
// 			name:     "2d 2",
// 			tensor:   MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:  []int{0, 1},
// 			expected: Scalar(2),
// 		},
// 		{
// 			name:      "too many index",
// 			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:   []int{0, 1, 1},
// 			expectErr: true,
// 		},
// 		{
// 			name:      "too big index",
// 			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
// 			indices:   []int{0, 3},
// 			expectErr: true,
// 		},
// 		{
// 			name:     "containing 0 in stride 1",
// 			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
// 			indices:  []int{0},
// 			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
// 		},
// 		{
// 			name:     "containing 0 in stride 2",
// 			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
// 			indices:  []int{1},
// 			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
// 		},
// 		{
// 			name:     "containing 0 in stride 3",
// 			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
// 			indices:  []int{1, 1},
// 			expected: MustNdShape(seq[float64](13, 25), 3, 4),
// 		},
// 		{
// 			name:     "containing 0 in stride 4",
// 			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
// 			indices:  []int{1, 1, 2},
// 			expected: MustNdShape(seq[float64](21, 25), 4),
// 		},
// 		{
// 			name:     "containing 0 in stride 5",
// 			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
// 			indices:  []int{1, 1, 2, 3},
// 			expected: Scalar(24),
// 		},
// 	}
//
// 	for _, tc := range tests {
// 		tc := tc
// 		t.Run(tc.name, func(t *testing.T) {
// 			t.Parallel()
// 			got, err := tc.tensor.Index(tc.indices...)
// 			checkErr(t, tc.expectErr, err)
// 			mustEq(t, tc.expected, got)
// 		})
// 	}
// }

// func TestIndex_Complicated(t *testing.T) {
// 	tensor := MustNdShape(seq[float64](1, 121), 2, 3, 4, 5)
//
// 	tensor2, err := tensor.Index(0)
// 	checkErr(t, false, err)
// 	mustEq(t, MustNdShape(seq[float64](1, 61), 3, 4, 5), tensor2)
//
// 	tensor3, err := tensor2.Index(2)
// 	checkErr(t, false, err)
// 	mustEq(t, MustNdShape(seq[float64](41, 61), 4, 5), tensor3)
//
// 	tensor4, err := tensor3.Index(1)
// 	checkErr(t, false, err)
// 	mustEq(t, Vector(seq[float64](46, 51)), tensor4)
//
// 	tensor5, err := tensor4.Index(4)
// 	checkErr(t, false, err)
// 	mustEq(t, Scalar(50), tensor5)
// }

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

func TestBool(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		f        func(f float64) bool
		expected *Tensor
	}{
		{
			name:     "scalar",
			tensor:   Scalar(3),
			f:        func(f float64) bool { return f < 0 },
			expected: Scalar(0),
		},
		{
			name:     "vector",
			tensor:   ArangeVec(1, 4, 1),
			f:        func(f float64) bool { return f < 2 },
			expected: Vector([]float64{1, 0, 0}),
		},
		{
			name:     "2d",
			tensor:   Must(ArangeVec(1, 7, 1).Reshape(2, 3)),
			f:        func(f float64) bool { return int(f)%2 == 0 },
			expected: Must(Vector([]float64{0, 1, 0, 1, 0, 1}).Reshape(2, 3)),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.tensor.Bool(tc.f)
			mustEq(t, tc.expected, got)
		})
	}
}
