package tensor2

import (
	"testing"
)

func TestIndex(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		args      []*IndexArg
		expected  *Tensor
		expectErr bool
	}{
		{
			name:     "vector 1",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{From(1)},
			expected: Vector([]float64{2, 3, 4, 5}),
		},
		{
			name:     "vector 2",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{To(2)},
			expected: Vector([]float64{1, 2}),
		},
		{
			name:     "2d 1",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{FromToBy(0, 4, 2), FromToBy(1, 4, 2)},
			expected: MustNdShape([]float64{2, 4, 10, 12}, 2, 2),
		},
		{
			name:     "2d 2",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{All(), FromToBy(1, 4, 2)},
			expected: MustNdShape([]float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}, 6, 2),
		},
		{
			name:     "3d 1",
			tensor:   MustNdShape(seq[float64](1, 25), 2, 3, 4),
			args:     []*IndexArg{FromTo(0, 2), FromToBy(0, 3, 2), To(2)},
			expected: MustNdShape([]float64{1, 2, 9, 10, 13, 14, 21, 22}, 2, 2, 2),
		},
		{
			name:     "3d 2",
			tensor:   MustNdShape(seq[float64](1, 25), 2, 3, 4),
			args:     []*IndexArg{FromTo(0, 2)},
			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
		},
		{
			name:     "containing 0 in stride 1",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(0)},
			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Index(tc.args...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestIndex_Complicated(t *testing.T) {
	tensor := MustNdShape(seq[float64](1, 121), 2, 3, 4, 5)

	tensor2, err := tensor.Index(At(0))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seq[float64](1, 61), 3, 4, 5), tensor2)

	tensor3, err := tensor2.Index(At(2))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seq[float64](41, 61), 4, 5), tensor3)

	tensor4, err := tensor3.Index(At(1))
	checkErr(t, false, err)
	mustEq(t, Vector(seq[float64](46, 51)), tensor4)

	tensor5, err := tensor4.Index(At(4))
	checkErr(t, false, err)
	mustEq(t, Scalar(50), tensor5)
}

func TestIndex_Complicated_slicing(t *testing.T) {
	tensor := MustNdShape(seq[float64](1, 121), 2, 3, 4, 5)

	tensor2, err := tensor.Index(From(1))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seq[float64](61, 121), 1, 3, 4, 5), tensor2)

	tensor3, err := tensor2.Index(All(), From(2))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seq[float64](101, 121), 1, 1, 4, 5), tensor3)

	tensor4, err := tensor3.Index(All(), All(), FromToBy(1, 4, 2))
	checkErr(t, false, err)
	mustEq(t, MustNdShape([]float64{106, 107, 108, 109, 110, 116, 117, 118, 119, 120}, 1, 1, 2, 5), tensor4)

	tensor5, err := tensor4.Index(All(), All(), All(), FromTo(2, 4))
	checkErr(t, false, err)
	mustEq(t, MustNdShape([]float64{108, 109, 118, 119}, 1, 1, 2, 2), tensor5)
}

func TestIndexWrite(t *testing.T) {
	// Add
	tensor := Vector([]float64{1, 2, 3, 4, 5})
	tensor.IndexAdd(2, At(1))
	mustEq(t, Vector([]float64{1, 4, 3, 4, 5}), tensor)

	// Sub
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexSub(2, At(0))
	mustEq(t, Must(New([][]float64{{-1, 0}, {3, 4}})), tensor)

	// Mul
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexMul(2, At(0))
	mustEq(t, Must(New([][]float64{{2, 4}, {3, 4}})), tensor)

	// Div
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexDiv(2, At(0))
	mustEq(t, Must(New([][]float64{{0.5, 1}, {3, 4}})), tensor)

	// Set
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexSet(2, At(0))
	mustEq(t, Must(New([][]float64{{2, 2}, {3, 4}})), tensor)
}
