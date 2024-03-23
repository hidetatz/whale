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
		{
			name:      "scalar",
			tensor:    Scalar(1),
			args:      []*IndexArg{},
			expectErr: true,
		},
		{
			name:      "missing index",
			tensor:    Vector([]float64{1, 2, 3}),
			args:      []*IndexArg{},
			expectErr: true,
		},
		{
			name:      "too many index",
			tensor:    Vector([]float64{1, 2, 3}),
			args:      []*IndexArg{At(0), At(1)},
			expectErr: true,
		},
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
			name:     "vector 3",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{FromTo(2, 4)},
			expected: Vector([]float64{3, 4}),
		},
		{
			name:     "vector 4",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{FromToBy(1, 5, 2)},
			expected: Vector([]float64{2, 4}),
		},
		{
			name:     "vector 5",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{By(2)},
			expected: Vector([]float64{1, 3, 5}),
		},
		{
			name:     "vector 6",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{From(-1)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 7",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{To(6)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 8",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{FromTo(-2, 8)},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 9",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{FromTo(6, 10)},
			expected: Vector([]float64{}),
		},
		{
			name:     "vector 10",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{FromTo(6, 1)},
			expected: Vector([]float64{}),
		},
		{
			name:     "vector 11",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{All()},
			expected: Vector([]float64{1, 2, 3, 4, 5}),
		},
		{
			name:     "vector 12",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{By(4)},
			expected: Vector([]float64{1, 5}),
		},
		{
			name:     "vector 13",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{By(5)},
			expected: Vector([]float64{1}),
		},
		{
			name:     "vector 14",
			tensor:   Vector([]float64{1, 2, 3, 4, 5}),
			args:     []*IndexArg{At(2)},
			expected: Scalar(3),
		},
		{
			name:      "step must not be 0",
			tensor:    Vector([]float64{1, 2, 3, 4, 5}),
			args:      []*IndexArg{By(0)},
			expectErr: true,
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
			name:     "2d 3",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{All()},
			expected: MustNdShape(seq[float64](1, 25), 6, 4),
		},
		{
			name:     "2d 4",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{All(), All()},
			expected: MustNdShape(seq[float64](1, 25), 6, 4),
		},
		{
			name:     "2d 5",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{From(6)},
			expected: MustNdShape([]float64{}, 0, 4),
		},
		{
			name:     "2d 6",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{At(4), At(2)},
			expected: Scalar(19),
		},
		{
			name:     "2d 7",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{At(4)},
			expected: Vector([]float64{17, 18, 19, 20}),
		},
		{
			name:      "2d 8",
			tensor:    MustNdShape(seq[float64](1, 25), 6, 4),
			args:      []*IndexArg{FromTo(2, 5), At(4)},
			expectErr: true,
		},
		{
			name:     "2d 8",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{FromTo(2, 5), At(3)},
			expected: Vector([]float64{12, 16, 20}),
		},
		{
			name:     "2d 9",
			tensor:   MustNdShape(seq[float64](1, 25), 6, 4),
			args:     []*IndexArg{FromToBy(2, 5, 2), At(3)},
			expected: Vector([]float64{12, 20}),
		},
		{
			name:      "too many index",
			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			args:      []*IndexArg{At(0), At(1), At(1)},
			expectErr: true,
		},
		{
			name:      "too big index",
			tensor:    MustNdShape([]float64{1, 2, 3, 4, 5, 6}, 2, 3),
			args:      []*IndexArg{At(0), At(3)},
			expectErr: true,
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
			name:     "3d 3",
			tensor:   MustNdShape(seq[float64](1, 25), 2, 3, 4),
			args:     []*IndexArg{FromTo(0, 2), At(0), To(2)},
			expected: MustNdShape([]float64{1, 2, 13, 14}, 2, 2),
		},
		{
			name:     "3d 4",
			tensor:   MustNdShape(seq[float64](1, 25), 2, 3, 4),
			args:     []*IndexArg{FromTo(0, 2), At(2), To(2)},
			expected: MustNdShape([]float64{9, 10, 21, 22}, 2, 2),
		},
		{
			name:     "containing 0 in stride 1",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(0)},
			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
		},
		{
			name:     "containing 0 in stride 2",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(1)},
			expected: MustNdShape(seq[float64](1, 25), 2, 3, 4),
		},
		{
			name:     "containing 0 in stride 3",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(1), At(1)},
			expected: MustNdShape(seq[float64](13, 25), 3, 4),
		},
		{
			name:     "containing 0 in stride 4",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(1), At(1), At(2)},
			expected: MustNdShape(seq[float64](21, 25), 4),
		},
		{
			name:     "containing 0 in stride 5",
			tensor:   &Tensor{data: seq[float64](1, 25), Shape: []int{2, 2, 3, 4}, Strides: []int{0, 12, 4, 1}},
			args:     []*IndexArg{At(1), At(1), At(2), At(3)},
			expected: Scalar(24),
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
