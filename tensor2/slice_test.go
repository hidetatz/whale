package tensor2

import "testing"

func TestSlice(t *testing.T) {
	t.Skip()
	tests := []struct {
		name      string
		tensor    *Tensor
		slices    []*Slice
		expected  *Tensor
		expectErr bool
	}{
		{
			name:      "scalar",
			tensor:    Scalar(1),
			slices:    []*Slice{},
			expectErr: true,
		},
		{
			name:     "vector 1",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{From(1)},
			expected: Vector([]float64{2, 3, 4, 5}),
		},
		{
			name:     "vector 2",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{To(2)},
			expected: Vector([]float64{1, 2}),
		},
		{
			name:     "vector 3",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{FromTo(2, 4)},
			expected: Vector([]float64{3, 4}),
		},
		{
			name:     "vector 4",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{FromToBy(1, 5, 2)},
			expected: Vector([]float64{2, 4}),
		},
		{
			name:     "vector 5",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{By(2)},
			expected: Vector([]float64{1, 3, 5}),
		},
		{
			name:     "vector 6",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{From(-1)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 7",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{To(6)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 8",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{FromTo(-2, 8)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 9",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{FromTo(6, 10)},
			expected: Vector([]float64{}),
		},
		{
			name:     "vector 10",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{FromTo(6, 1)},
			expected: Vector([]float64{}),
		},
		{
			name:     "vector 11",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{All()},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 12",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{By(4)},
			expected: Vector([]float64{1, 5}),
		},
		{
			name:     "vector 13",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			slices:   []*Slice{By(5)},
			expected: Vector([]float64{1}),
		},
		{
			name:      "step must not be 0",
			tensor:    Vector([]float64{1, 2, 3, 4, 5}),
			slices:    []*Slice{By(0)},
			expectErr: true,
		},
		{
			name:     "2d 1",
			tensor:   MustNdShape(seqf(1, 25), 6, 4),
			slices:   []*Slice{FromToBy(0, 4, 2), FromToBy(1, 4, 2)},
			expected: MustNdShape([]float64{2, 4, 10, 12}, 2, 2),
		},
		{
			name:     "2d 2",
			tensor:   MustNdShape(seqf(1, 25), 6, 4),
			slices:   []*Slice{All(), FromToBy(1, 4, 2)},
			expected: MustNdShape([]float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}, 6, 2),
		},
		{
			name:     "2d 3",
			tensor:   MustNdShape(seqf(1, 25), 6, 4),
			slices:   []*Slice{All()},
			expected: MustNdShape(seqf(1, 25), 6, 4),
		},
		{
			name:     "2d 4",
			tensor:   MustNdShape(seqf(1, 25), 6, 4),
			slices:   []*Slice{All(), All()},
			expected: MustNdShape(seqf(1, 25), 6, 4),
		},
		{
			name:     "2d 5",
			tensor:   MustNdShape(seqf(1, 25), 6, 4),
			slices:   []*Slice{From(6)},
			expected: MustNdShape([]float64{}, 0, 4),
		},
		{
			name:     "3d 1",
			tensor:   MustNdShape(seqf(1, 25), 2, 3, 4),
			slices:   []*Slice{FromTo(0, 2), FromToBy(0, 3, 2), To(2)},
			expected: MustNdShape([]float64{1, 2, 9, 10, 13, 14, 21, 22}, 2, 2, 2),
		},
		{
			name:     "3d 2",
			tensor:   MustNdShape(seqf(1, 25), 2, 3, 4),
			slices:   []*Slice{FromTo(0, 2)},
			expected: MustNdShape(seqf(1, 25), 2, 3, 4),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := tc.tensor.Slice(tc.slices...)
			checkErr(t, tc.expectErr, err)
			mustEq(t, tc.expected, got)
		})
	}
}

func TestSlice_Complicated(t *testing.T) {
	tensor := MustNdShape(seqf(1, 121), 2, 3, 4, 5)

	tensor2, err := tensor.Slice(From(1))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seqf(61, 121), 1, 3, 4, 5), tensor2)

	tensor3, err := tensor2.Slice(All(), From(2))
	checkErr(t, false, err)
	mustEq(t, MustNdShape(seqf(101, 121), 1, 1, 4, 5), tensor3)

	tensor4, err := tensor3.Slice(All(), All(), FromToBy(1, 4, 2))
	checkErr(t, false, err)
	mustEq(t, MustNdShape([]float64{106, 107, 108, 109, 110, 116, 117, 118, 119, 120}, 1, 1, 2, 5), tensor4)

	tensor5, err := tensor4.Slice(All(), All(), All(), FromTo(2, 4))
	checkErr(t, false, err)
	mustEq(t, MustNdShape([]float64{108, 109, 118, 119}, 1, 1, 2, 2), tensor5)
}
