package main

import (
	"fmt"
	"math"
)

type Tensor struct {
}

type Variable struct {
	data    float64
	grad    *float64
	creator Function
}

func NewVar(data float64) *Variable {
	return &Variable{data: data, grad: nil}
}

func (v *Variable) String() string {
	return fmt.Sprintf("%v", v.data)
}

func (v *Variable) ClearGrad() {
	v.grad = nil
}

func (v *Variable) Backward() {
	if v.grad == nil {
		g := 1.0
		v.grad = &g
	}
	fs := []Function{v.creator}
	for len(fs) > 0 {
		var last Function
		last, fs = fs[len(fs)-1], fs[:len(fs)-1] // pop last

		ys := []float64{}
		for _, o := range last.GetOutputs() {
			ys = append(ys, *o.grad)
		}

		gxs := last.Backward(ys)
		for i, x := range last.GetInputs() {
			gx := gxs[i]
			if x.grad == nil {
				x.grad = &gx
			} else {
				s := *x.grad + gx
				x.grad = &s
			}

			if x.creator != nil {
				fs = append(fs, x.creator)
			}
		}

	}
}

type Function interface {
	Forward(inputs []*Variable) []*Variable
	GetInputs() []*Variable
	GetOutputs() []*Variable
	Backward(inputs []float64) []float64
}

func square(v *Variable) *Variable {
	s := &Square{}
	return s.Forward([]*Variable{v})[0]
}

type Square struct {
	inputs  []*Variable
	outputs []*Variable
}

func (s *Square) Forward(inputs []*Variable) []*Variable {
	s.inputs = inputs
	v := NewVar(math.Pow(inputs[0].data, 2))
	v.creator = s
	s.outputs = []*Variable{v}
	return s.outputs
}

func (s *Square) GetInputs() []*Variable {
	return s.inputs
}

func (s *Square) GetOutputs() []*Variable {
	return s.outputs
}

func (s *Square) Backward(gy []float64) []float64 {
	return []float64{2 * s.inputs[0].data * gy[0]}
}

func exp(v *Variable) *Variable {
	e := &Exp{}
	return e.Forward([]*Variable{v})[0]
}

type Exp struct {
	inputs  []*Variable
	outputs []*Variable
}

func (e *Exp) Forward(inputs []*Variable) []*Variable {
	e.inputs = inputs
	v := NewVar(math.Exp(inputs[0].data))
	v.creator = e
	e.outputs = []*Variable{v}
	return e.outputs
}

func (e *Exp) GetInputs() []*Variable {
	return e.inputs
}

func (e *Exp) GetOutputs() []*Variable {
	return e.outputs
}

func (e *Exp) Backward(gy []float64) []float64 {
	return []float64{math.Exp(e.inputs[0].data) * gy[0]}
}

func add(x1, x2 *Variable) *Variable {
	a := &Add{}
	return a.Forward([]*Variable{x1, x2})[0]
}

type Add struct {
	inputs  []*Variable
	outputs []*Variable
}

func (a *Add) Forward(inputs []*Variable) []*Variable {
	a.inputs = inputs
	v := NewVar(inputs[0].data + inputs[1].data)
	v.creator = a
	a.outputs = []*Variable{v}
	return a.outputs
}

func (a *Add) GetInputs() []*Variable {
	return a.inputs
}

func (a *Add) GetOutputs() []*Variable {
	return a.outputs
}

func (a *Add) Backward(gy []float64) []float64 {
	return []float64{gy[0], gy[0]}
}

func numDiff(f func(*Variable) *Variable, x *Variable, epsilon float64) float64 {
	x0 := NewVar(x.data - epsilon)
	x1 := NewVar(x.data + epsilon)
	y0 := f(x0)
	y1 := f(x1)
	return (y1.data - y0.data) / (2 * epsilon)
}

func main() {
	// x := NewVar(0.5)
	// a := square(x)
	// b := exp(a)
	// y := square(b)

	// y.Backward()

	// fmt.Println(*x.grad)

	// x2 := NewVar(2)
	// x3 := NewVar(3)
	// y2 := add(square(x2), square(x3))
	// y2.Backward()

	// fmt.Println(y2)
	// fmt.Println(*x2.grad)
	// fmt.Println(*x3.grad)

	x := NewVar(3)
	y := add(x, x)
	y.Backward()

	fmt.Println(*x.grad)

	x.ClearGrad()
	y = add(add(x, x), x)
	y.Backward()

	fmt.Println(*x.grad)
}
