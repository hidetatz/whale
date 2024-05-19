package tensor

import "testing"

func TestUniversalfunc1_Do(t *testing.T) {
	tests := []struct {
		name      string
		x         *Tensor
		target    *Tensor
		ufunc     *universalfunc1
		expected  *Tensor
		expectErr bool
	}{
		{
			name:  "exp",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: EXP,
			expected: New([][]float64{
				{2.718281828459045, 7.38905609893065, 20.085536923187668},
				{54.598150033144236, 148.4131591025766, 403.4287934927351},
			}),
		},
		{
			name:  "neg",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: NEG,
			expected: New([][]float64{
				{-1, -2, -3},
				{-4, -5, -6},
			}),
		},
		{
			name:  "sin",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: SIN,
			expected: New([][]float64{
				{0.8414709848078965, 0.9092974268256816, 0.1411200080598672},
				{-0.7568024953079282, -0.9589242746631385, -0.27941549819892586},
			}),
		},
		{
			name:  "cos",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: COS,
			expected: New([][]float64{
				{0.5403023058681398, -0.4161468365471424, -0.9899924966004454},
				{-0.6536436208636119, 0.2836621854632263, 0.9601702866503661},
			}),
		},
		{
			name:  "tanh",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: TANH,
			expected: New([][]float64{
				{0.7615941559557649, 0.9640275800758169, 0.9950547536867305},
				{0.999329299739067, 0.9999092042625951, 0.9999877116507956},
			}),
		},
		{
			name:  "log",
			x:     Vector([]float64{1, 2, 3, 4, 5, 6}).Reshape(2, 3),
			ufunc: LOG,
			expected: New([][]float64{
				{0, 0.6931471805599453, 1.0986122886681096},
				{1.3862943611198906, 1.6094379124341003, 1.791759469228055},
			}),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			mustEq(t, tc.expected, tc.ufunc.Do(tc.x))
		})
	}
}
