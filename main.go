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

func (v *Variable) Backward() {
	if v.grad == nil {
		g := 1.0
		v.grad = &g
	}
	fs := []Function{v.creator}
	for len(fs) > 0 {
		var last Function
		last, fs = fs[len(fs)-1], fs[:len(fs)-1] // pop last
		x, y := last.GetInput(), last.GetOutput()
		grad := last.Backward(*y.grad)
		x.grad = &grad
		if x.creator != nil {
			fs = append(fs, x.creator)
		}
	}
}

type Function interface {
	Forward(input *Variable) *Variable
	GetInput() *Variable
	GetOutput() *Variable
	Backward(input float64) float64
}

func square(v *Variable) *Variable {
	s := &Square{}
	return s.Forward(v)
}

type Square struct {
	input  *Variable
	output *Variable
}

func (s *Square) Forward(input *Variable) *Variable {
	s.input = input
	v := NewVar(math.Pow(input.data, 2))
	v.creator = s
	s.output = v
	return v
}

func (s *Square) GetInput() *Variable {
	return s.input
}

func (s *Square) GetOutput() *Variable {
	return s.output
}

func (s *Square) Backward(gy float64) float64 {
	return 2 * s.input.data * gy
}

func exp(v *Variable) *Variable {
	e := &Exp{}
	return e.Forward(v)
}

type Exp struct {
	input  *Variable
	output *Variable
}

func (e *Exp) Forward(input *Variable) *Variable {
	e.input = input
	v := NewVar(math.Exp(input.data))
	v.creator = e
	e.output = v
	return v
}

func (e *Exp) GetInput() *Variable {
	return e.input
}

func (e *Exp) GetOutput() *Variable {
	return e.output
}

func (e *Exp) Backward(gy float64) float64 {
	return math.Exp(e.input.data) * gy
}

func numDiff(f func(*Variable) *Variable, x *Variable, epsilon float64) float64 {
	x0 := NewVar(x.data - epsilon)
	x1 := NewVar(x.data + epsilon)
	y0 := f(x0)
	y1 := f(x1)
	return (y1.data - y0.data) / (2 * epsilon)
}

func main() {
	x := NewVar(0.5)
	a := square(x)
	b := exp(a)
	y := square(b)

	y.Backward()

	fmt.Println(*x.grad)
}
