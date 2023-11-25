package whale

import (
	"math"
	"testing"
)

func TestGrad(t *testing.T) {
	tests := []struct {
		name string
		f    func([]*Variable) []*Variable
		x    []*Variable
	}{
		{
			name: "square",
			f:    func(x []*Variable) []*Variable { return []*Variable{square(x[0])} },
			x:    []*Variable{NewVar(2)},
		},
		{
			name: "exp",
			f:    func(x []*Variable) []*Variable { return []*Variable{exp(x[0])} },
			x:    []*Variable{NewVar(2)},
		},
		{
			name: "add",
			f:    func(x []*Variable) []*Variable { return []*Variable{add(x[0], x[1])} },
			x:    []*Variable{NewVar(2), NewVar(3)},
		},
		{
			name: "sub",
			f:    func(x []*Variable) []*Variable { return []*Variable{sub(x[0], x[1])} },
			x:    []*Variable{NewVar(2), NewVar(3)},
		},
		{
			name: "mul",
			f:    func(x []*Variable) []*Variable { return []*Variable{mul(x[0], x[1])} },
			x:    []*Variable{NewVar(2), NewVar(3)},
		},
		{
			name: "div",
			f:    func(x []*Variable) []*Variable { return []*Variable{div(x[0], x[1])} },
			x:    []*Variable{NewVar(2), NewVar(3)},
		},
		{
			name: "neg",
			f:    func(x []*Variable) []*Variable { return []*Variable{neg(x[0])} },
			x:    []*Variable{NewVar(2)},
		},
		{
			name: "pow",
			// pow (a^b) requires a and b but grad is calculated on only 1 value (a).
			// If b is included in x, then this test code fails when calculating diff on b,
			// so b is not included in x.
			f: func(x []*Variable) []*Variable { return []*Variable{pow(x[0], NewVar(3))} },
			x: []*Variable{NewVar(2)},
		},
		{
			name: "combined 1",
			f: func(x []*Variable) []*Variable {
				a := add(x[0], x[1])
				b := mul(a, x[2])
				y := div(b, x[3])
				return []*Variable{y}
			},
			x: []*Variable{NewVar(2), NewVar(3), NewVar(4), NewVar(5)},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			y := tc.f(tc.x)[0]
			y.Backward()

			grad := numDiff(t, tc.f, tc.x)
			for i := range grad {
				delta := math.Abs(*tc.x[i].grad - grad[i])
				if delta > 1e-4 {
					t.Errorf("backward: %v, numerical diff: %v", *tc.x[i].grad, grad[i])
				}
			}
		})
	}
}

// numDiff differentiates x with f by numerical propagation for testing purpose
func numDiff(t *testing.T, f func([]*Variable) []*Variable, xs []*Variable) []float64 {
	h := 1e-4
	grad := []float64{}

	// Do partial differentiation by changing only one x at a time.
	// We use this formula for each x: (f(x1 + h, x2, ...)  - f(x1 - h, x2, ...)) / 2h
	for i := range xs {
		// save current focusing x to restore later.
		tmp := xs[i]

		// First calculate f(x1 + h, x2, ...)
		xs[i].data += h
		y1 := f(xs)

		// Second, calculate f(x1 - h, x2, ...)
		// h * 2 is required because xs[i].data is already added above
		xs[i].data -= h * 2
		y2 := f(xs)

		// At last do (y1 - y2) / 2h. This is the gradient
		grad = append(grad, (y1[0].data-y2[0].data)/(2*h))

		// restore the original x
		xs[i] = tmp
	}

	return grad
}
