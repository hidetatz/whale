package tensor2

import "testing"

func Test_advancedIndex(t *testing.T) {
	tests := []struct {
		name      string
		tensor    *Tensor
		args      []*IndexArg
		expected  *Tensor
		expectErr bool
	}{
		{
			name:     "vector 1",
			tensor:   ArangeVec(0, 6, 1),
			args:     []*IndexArg{List(ArangeVec(2, 5, 1))},
			expected: ArangeVec(2, 5, 1),
		},
		{
			name:     "vector 2",
			tensor:   ArangeVec(0, 6, 1),
			args:     []*IndexArg{List(MustNew([][]float64{{2, 3, 5}, {0, 1, 4}}))},
			expected: MustNew([][]float64{{2, 3, 5}, {0, 1, 4}}),
		},
		{
			name:   "too many indices",
			tensor: ArangeVec(0, 6, 1),
			args: []*IndexArg{
				List(MustNew([][]float64{{2, 3, 5}, {0, 1, 4}})),
				List(MustNew([][]float64{{2, 3, 5}, {0, 1, 4}})),
			},
			expectErr: true,
		},
		{
			name:   "2d single list",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(Vector([]float64{0, 1, 0})),
			},
			expected: MustNew([][]float64{
				{0, 1, 2},
				{3, 4, 5},
				{0, 1, 2},
			}),
		},
		{
			name:   "2d multiple lists",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(Vector([]float64{0, 1, 0})),
				List(Vector([]float64{1, 0, 2})),
			},
			expected: Vector([]float64{1, 3, 2}),
		},
		{
			name:   "2d index on 2d",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
			},
			expected: MustNew([][][]float64{
				{
					{0, 1, 2},
					{3, 4, 5},
					{0, 1, 2},
				},
				{
					{3, 4, 5},
					{0, 1, 2},
					{3, 4, 5},
				},
			}),
		},
		{
			name:   "2d broadcast",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
				List(Scalar(2)),
			},
			expected: MustNew([][]float64{
				{2, 5, 2},
				{5, 2, 5},
			}),
		},
		{
			name:   "invalid broadcast",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
				List(Vector([]float64{1, 0})),
			},
			expectErr: true,
		},
		{
			name:   "slice combined",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				FromTo(0, 2),
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
			},
			expected: MustNew([][][]float64{
				{
					{0, 1, 0},
					{1, 0, 1},
				},
				{
					{3, 4, 3},
					{4, 3, 4},
				},
			}),
		},
		{
			name:   "slice follows2",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				FromToBy(1, 2, 2),
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
			},
			expected: MustNew([][][]float64{
				{
					{3, 4, 3},
					{4, 3, 4},
				},
			}),
		},
		{
			name:   "slice follows",
			tensor: Must(Arange(0, 6, 1, 2, 3)),
			args: []*IndexArg{
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
				FromTo(0, 2),
			},
			expected: MustNew([][][]float64{
				{
					{0, 1},
					{3, 4},
					{0, 1},
				},
				{
					{3, 4},
					{0, 1},
					{3, 4},
				},
			}),
		},
		{
			name:   "separated",
			tensor: Must(Arange(0, 48, 1, 2, 3, 4, 2)),
			args: []*IndexArg{
				At(1),
				FromTo(1, 2),
				List(MustNew([][]float64{{0, 1, 0}, {1, 0, 1}})),
			},
			expected: MustNew([][][][]float64{
				{

					{
						{32, 33},
					},
					{
						{34, 35},
					},
					{
						{32, 33},
					},
				},
				{
					{
						{34, 35},
					},
					{
						{32, 33},
					},
					{
						{34, 35},
					},
				},
			}),
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
