package whale

import (
	"math"
)

type Operation interface {
	Forward(inputs ...*Variable) []*Variable
	Backward(gy ...float64) []float64
}

/*
 * Square
 */

func square(v *Variable) *Variable {
	f := NewFunction(&Square{})
	return f.forward(v)[0]
}

type Square struct {
	input *Variable
}

func (s *Square) Forward(inputs ...*Variable) []*Variable {
	s.input = inputs[0]
	v := NewVar(math.Pow(s.input.data, 2))
	out := []*Variable{v}
	return out
}

func (s *Square) Backward(gy ...float64) []float64 {
	return []float64{2 * s.input.data * gy[0]}
}

/*
 * Exp
 */

func exp(v *Variable) *Variable {
	f := NewFunction(&Exp{})
	return f.forward(v)[0]
}

type Exp struct {
	input *Variable
}

func (e *Exp) Forward(inputs ...*Variable) []*Variable {
	e.input = inputs[0]
	v := NewVar(math.Exp(e.input.data))
	out := []*Variable{v}
	return out
}

func (e *Exp) Backward(gy ...float64) []float64 {
	return []float64{math.Exp(e.input.data) * gy[0]}
}

/*
 * Add, Sub, Mul, Div, Pow
 */

func add(x1, x2 *Variable) *Variable {
	f := NewFunction(&Add{})
	return f.forward(x1, x2)[0]
}

type Add struct {
	inputs []*Variable
}

func (a *Add) Forward(inputs ...*Variable) []*Variable {
	a.inputs = inputs
	v := NewVar(inputs[0].data + inputs[1].data)
	out := []*Variable{v}
	return out
}

func (a *Add) Backward(gy ...float64) []float64 {
	return []float64{gy[0], gy[0]}
}

func sub(x1, x2 *Variable) *Variable {
	f := NewFunction(&Sub{})
	return f.forward(x1, x2)[0]
}

type Sub struct {
	inputs []*Variable
}

func (s *Sub) Forward(inputs ...*Variable) []*Variable {
	s.inputs = inputs
	v := NewVar(inputs[0].data - inputs[1].data)
	out := []*Variable{v}
	return out
}

func (s *Sub) Backward(gy ...float64) []float64 {
	return []float64{gy[0], -gy[0]}
}

func mul(x1, x2 *Variable) *Variable {
	f := NewFunction(&Mul{})
	return f.forward(x1, x2)[0]
}

type Mul struct {
	inputs []*Variable
}

func (m *Mul) Forward(inputs ...*Variable) []*Variable {
	m.inputs = inputs
	v := NewVar(inputs[0].data * inputs[1].data)
	out := []*Variable{v}
	return out
}

func (m *Mul) Backward(gy ...float64) []float64 {
	x0, x1 := m.inputs[0].data, m.inputs[1].data
	return []float64{gy[0] * x1, gy[0] * x0}
}

func div(x1, x2 *Variable) *Variable {
	f := NewFunction(&Div{})
	return f.forward(x1, x2)[0]
}

type Div struct {
	inputs []*Variable
}

func (d *Div) Forward(inputs ...*Variable) []*Variable {
	d.inputs = inputs
	v := NewVar(inputs[0].data / inputs[1].data)
	out := []*Variable{v}
	return out
}

func (d *Div) Backward(gy ...float64) []float64 {
	x0, x1 := d.inputs[0].data, d.inputs[1].data
	return []float64{gy[0] / x1, gy[0] * (-x0 / math.Pow(x1, 2))}
}

func neg(x *Variable) *Variable {
	f := NewFunction(&Neg{})
	return f.forward(x)[0]
}

type Neg struct {
	input *Variable
}

func (n *Neg) Forward(inputs ...*Variable) []*Variable {
	n.input = inputs[0]
	v := NewVar(-inputs[0].data)
	out := []*Variable{v}
	return out
}

func (n *Neg) Backward(gy ...float64) []float64 {
	return []float64{-gy[0]}
}

func pow(x, c *Variable) *Variable {
	f := NewFunction(&Pow{c: c})
	return f.forward(x)[0]
}

type Pow struct {
	c     *Variable
	input *Variable
}

func (p *Pow) Forward(inputs ...*Variable) []*Variable {
	p.input = inputs[0]
	v := NewVar(math.Pow(inputs[0].data, p.c.data))
	out := []*Variable{v}
	return out
}

func (p *Pow) Backward(gy ...float64) []float64 {
	x, c := p.input.data, p.c.data
	return []float64{c * math.Pow(x, c-1) * gy[0]}
}
