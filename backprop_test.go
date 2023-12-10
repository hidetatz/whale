package whale

import (
	// "fmt"
	"math"
	"testing"

	"github.com/hidetatz/whale/tensor"
)

func TestGrad(t *testing.T) {
	tests := []struct {
		name string
		f    func([]*Variable) []*Variable
		x    []*Variable
	}{
		{
			name: "square",
			f:    func(x []*Variable) []*Variable { return []*Variable{Square_(x[0])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2))},
		},
		{
			name: "exp",
			f:    func(x []*Variable) []*Variable { return []*Variable{Exp_(x[0])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2))},
		},
		{
			name: "add",
			f:    func(x []*Variable) []*Variable { return []*Variable{Add_(x[0], x[1])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2)), NewVar(tensor.FromScalar(3))},
		},
		{
			name: "sub",
			f:    func(x []*Variable) []*Variable { return []*Variable{Sub_(x[0], x[1])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2)), NewVar(tensor.FromScalar(3))},
		},
		{
			name: "mul",
			f:    func(x []*Variable) []*Variable { return []*Variable{Mul_(x[0], x[1])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2)), NewVar(tensor.FromScalar(3))},
		},
		{
			name: "div",
			f:    func(x []*Variable) []*Variable { return []*Variable{Div_(x[0], x[1])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2)), NewVar(tensor.FromScalar(3))},
		},
		{
			name: "neg",
			f:    func(x []*Variable) []*Variable { return []*Variable{Neg_(x[0])} },
			x:    []*Variable{NewVar(tensor.FromScalar(2))},
		},
		{
			name: "pow",
			// pow (a^b) requires a and b but grad is calculated only for a.
			// If b is included in x below, then this test code fails when calculating diff on b,
			// so b is directly passed to pow(), instead of being included in x.
			f: func(x []*Variable) []*Variable { return []*Variable{Pow_(x[0], NewVar(tensor.FromScalar(3)))} },
			x: []*Variable{NewVar(tensor.FromScalar(2))},
		},
		{
			name: "combined 1",
			f: func(x []*Variable) []*Variable {
				a := Add_(x[0], x[1])
				b := Mul_(a, x[2])
				y := Div_(b, x[3])
				return []*Variable{y}
			},
			x: []*Variable{NewVar(tensor.FromScalar(2)), NewVar(tensor.FromScalar(3)), NewVar(tensor.FromScalar(4)), NewVar(tensor.FromScalar(5))},
		},
		{
			name: "sphere",
			f: func(x []*Variable) []*Variable {
				y := Add_(Square_(x[0]), Square_(x[1]))
				return []*Variable{y}
			},
			x: []*Variable{NewVar(tensor.FromScalar(1)), NewVar(tensor.FromScalar(1))},
		},
		{
			name: "matyas",
			f: func(x []*Variable) []*Variable {
				t1 := Pow_(x[0], NewVar(tensor.FromScalar(2)))
				t2 := Pow_(x[1], NewVar(tensor.FromScalar(2)))
				t3 := Mul_(NewVar(tensor.FromScalar(0.26)), Add_(t1, t2))
				t4 := Mul_(NewVar(tensor.FromScalar(0.48)), x[0])
				t5 := Mul_(t4, x[1])
				y := Sub_(t3, t5)
				return []*Variable{y}
			},
			x: []*Variable{NewVar(tensor.FromScalar(1)), NewVar(tensor.FromScalar(1))},
		},
		// {
		// 	name: "matyas",
		// 	f: "0.26 * (x[0] ** 2 + x[1] ++ 2) - 0.48 * x[0] * x[1]"
		// 	x: []*Variable{NewVar(tensor.FromScalar(1)), NewVar(tensor.FromScalar(1))},
		// },
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			y := tc.f(tc.x)[0]
			y.Backward()

			grad := numDiff(t, tc.f, tc.x)
			for i := range grad {
				// fmt.Println(tc.x[i].grad.data.Data)
				// fmt.Println(grad[i].Data)
				delta := math.Abs(Sub_(tc.x[i].grad, NewVar(grad[i])).data.Data[0])
				if delta > 1e-4 {
					t.Errorf("backward: %v, numerical diff: %v", tc.x[i].grad.data.Data[0], grad[i].Data[0])
				}
			}
		})
	}
}

// numDiff differentiates x with f by numerical propagation for testing purpose
func numDiff(t *testing.T, f func([]*Variable) []*Variable, xs []*Variable) []*tensor.Tensor {
	t.Helper()
	h := 1e-4
	grad := []*tensor.Tensor{}

	// Do partial differentiation by changing only one x at a time.
	// We use this formula for each x: (f(x1 + h, x2, ...)  - f(x1 - h, x2, ...)) / 2h
	for i := range xs {
		// save current focusing x to restore later.
		tmp := xs[i]

		// First calculate f(x1 + h, x2, ...)
		xs[i].data = device.Add(xs[i].data, tensor.FromScalar(h))
		y1 := f(xs)

		// Second, calculate f(x1 - h, x2, ...)
		// h * 2 is required because xs[i].data is already added above
		xs[i].data = device.Sub(xs[i].data, tensor.FromScalar(h*2))
		y2 := f(xs)

		// At last do (y1 - y2) / 2h. This is the gradient
		grad = append(grad, device.Div(device.Sub(y1[0].data, y2[0].data), tensor.FromScalar(2*h)))

		// restore the original x
		xs[i] = tmp
	}

	return grad
}

func TestHigherOrderBackprop(t *testing.T) {
	f := func(x []*Variable) []*Variable {
		t1 := Pow_(x[0], NewVar(tensor.FromScalar(4)))
		t2 := Pow_(x[0], NewVar(tensor.FromScalar(2)))
		t3 := Mul_(NewVar(tensor.FromScalar(2)), t2)
		y := Sub_(t1, t3)
		return []*Variable{y}
	}
	x := []*Variable{NewVar(tensor.FromScalar(2))}

	for i := 0; i < 10; i++ {
		y := f(x)[0]
		x[0].ClearGrad()
		y.Backward()

		gx := x[0].GetGrad()
		x[0].ClearGrad()
		gx.Backward()
		gx2 := x[0].GetGrad()

		x[0].SetData(device.Sub(x[0].GetData(), device.Div(gx.GetData(), gx2.GetData())))
	}

	if x[0].GetData().Data[0] != 1 {
		t.Errorf("failed!")
	}
}
