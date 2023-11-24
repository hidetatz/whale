package main

import (
	"fmt"
	"math"
	"sort"
)

var EnableBackprop = true

type Variable struct {
	data    float64
	grad    *float64
	creator Function
	generation int
}

func NewVar(data float64) *Variable {
	return &Variable{data: data}
}

func (v *Variable) String() string {
	return fmt.Sprintf("%v", v.data)
}

func (v *Variable) ClearGrad() {
	v.grad = nil
}

func (v *Variable) SetCreator(creator Function) {
	v.creator = creator
	v.generation = creator.GetGeneration() + 1
}

func (v *Variable) Backward() {
	if v.grad == nil {
		g := 1.0
		v.grad = &g
	}

	fs := []Function{}
	uniqueadd := func(f Function) {
		for _, added := range fs {
			if added == f {
				return
			}
		}
		// not added
		fs = append(fs, f)
		sort.Slice(fs, func(i, j int) bool {
			return fs[i].GetGeneration() < fs[j].GetGeneration()
		})
	}

	uniqueadd(v.creator)

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
				uniqueadd(x.creator)
			}
		}

	}
}

type Function interface {
	Forward(inputs []*Variable) []*Variable
	GetInputs() []*Variable
	GetOutputs() []*Variable
	GetGeneration() int
	Backward(inputs []float64) []float64
}

func square(v *Variable) *Variable {
	s := &Square{}
	return s.Forward([]*Variable{v})[0]
}

func getMaxGen(vs []*Variable) int {
	max := 0
	for _, v := range vs {
		if v.generation > max {
			max = v.generation
		}
	}
	return max
}

type Square struct {
	inputs  []*Variable
	outputs []*Variable
	generation int
}

func (s *Square) Forward(inputs []*Variable) []*Variable {
	v := NewVar(math.Pow(inputs[0].data, 2))
	out := []*Variable{v}
	if EnableBackprop {
		s.inputs = inputs
		v.SetCreator(s)
		s.generation = getMaxGen(inputs)
		s.outputs = out
	}
	return out
}

func (s *Square) GetInputs() []*Variable {
	return s.inputs
}

func (s *Square) GetOutputs() []*Variable {
	return s.outputs
}

func (s *Square) GetGeneration() int {
	return s.generation
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
	generation int
}

func (e *Exp) Forward(inputs []*Variable) []*Variable {
	v := NewVar(math.Exp(inputs[0].data))
	out := []*Variable{v}
	if EnableBackprop {
		e.inputs = inputs
		v.SetCreator(e)
		e.generation = getMaxGen(inputs)
		e.outputs = out
	}
	return out
}

func (e *Exp) GetInputs() []*Variable {
	return e.inputs
}

func (e *Exp) GetOutputs() []*Variable {
	return e.outputs
}

func (e *Exp) GetGeneration() int {
	return e.generation
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
	generation int
}

func (a *Add) Forward(inputs []*Variable) []*Variable {
	v := NewVar(inputs[0].data + inputs[1].data)
	out := []*Variable{v}
	if EnableBackprop {
		a.inputs = inputs
		v.SetCreator(a)
		a.generation = getMaxGen(inputs)
		a.outputs = out
	}
	return out
}

func (a *Add) GetInputs() []*Variable {
	return a.inputs
}

func (a *Add) GetOutputs() []*Variable {
	return a.outputs
}

func (a *Add) GetGeneration() int {
	return a.generation
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

	x := NewVar(2)
	a := square(x)
	y := add(square(a), square(a))
	y.Backward()

	fmt.Println(y, *x.grad)

	// x.ClearGrad()
	// y = add(add(x, x), x)
	// y.Backward()

	// fmt.Println(*x.grad)
}
