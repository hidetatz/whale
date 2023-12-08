package whale

import (
	"fmt"
	"math"

	"github.com/hidetatz/whale/tensor"
)

var device Device

func init() {
	device = &CPU{}
}

type Operation interface {
	fmt.Stringer
	Forward(inputs ...*Variable) []*Variable
	Backward(gy ...*Variable) []*Variable
}

/*
 * Square
 */

func Square_(v *Variable) *Variable {
	f := NewFunction(&Square{})
	return f.forward(v)[0]
}

type Square struct {
	input *Variable
}

func (s *Square) Forward(inputs ...*Variable) []*Variable {
	s.input = inputs[0]
	v := NewVar(device.Pow(s.input.data, 2))
	out := []*Variable{v}
	return out
}

func (s *Square) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Mul_(Mul_(NewVar(tensor.FromScalar(2)), s.input), gy[0])}
}

func (s *Square) String() string {
	return "^2"
}

/*
 * Exp
 */

func Exp_(v *Variable) *Variable {
	f := NewFunction(&Exp{})
	return f.forward(v)[0]
}

type Exp struct {
	input  *Variable
	output *Variable
}

func (e *Exp) Forward(inputs ...*Variable) []*Variable {
	e.input = inputs[0]
	v := NewVar(device.Exp(e.input.data))
	out := []*Variable{v}
	e.output = v
	return out
}

func (e *Exp) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Mul_(e.output, gy[0])}
}

func (e *Exp) String() string {
	return "e^x"
}

/*
 * Add, Sub, Mul, Div, Pow
 */

func Add_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Add{})
	return f.forward(x1, x2)[0]
}

type Add struct {
	inputs []*Variable
}

func (a *Add) Forward(inputs ...*Variable) []*Variable {
	a.inputs = inputs
	v := NewVar(device.Add(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (a *Add) Backward(gy ...*Variable) []*Variable {
	return []*Variable{gy[0], gy[0]}
}

func (a *Add) String() string {
	return "+"
}

func Sub_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Sub{})
	return f.forward(x1, x2)[0]
}

type Sub struct {
	inputs []*Variable
}

func (s *Sub) Forward(inputs ...*Variable) []*Variable {
	s.inputs = inputs
	v := NewVar(device.Sub(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (s *Sub) Backward(gy ...*Variable) []*Variable {
	return []*Variable{gy[0], Neg_(gy[0])}
}

func (s *Sub) String() string {
	return "-"
}

func Mul_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Mul{})
	return f.forward(x1, x2)[0]
}

type Mul struct {
	inputs []*Variable
}

func (m *Mul) Forward(inputs ...*Variable) []*Variable {
	m.inputs = inputs
	v := NewVar(device.Mul(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (m *Mul) Backward(gy ...*Variable) []*Variable {
	x0, x1 := m.inputs[0], m.inputs[1]
	return []*Variable{Mul_(gy[0], x1), Mul_(gy[0], x0)}
}

func (m *Mul) String() string {
	return "*"
}

func Div_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Div{})
	return f.forward(x1, x2)[0]
}

type Div struct {
	inputs []*Variable
}

func (d *Div) Forward(inputs ...*Variable) []*Variable {
	d.inputs = inputs
	v := NewVar(device.Div(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (d *Div) Backward(gy ...*Variable) []*Variable {
	x0, x1 := d.inputs[0], d.inputs[1]
	return []*Variable{Div_(gy[0], x1), Mul_(gy[0], Div_(Neg_(x0), Pow_(x1, NewVar(2))))}
}

func (d *Div) String() string {
	return "/"
}

func Neg_(x *Variable) *Variable {
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

func (n *Neg) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Neg_(gy[0])}
}

func (n *Neg) String() string {
	return "-x"
}

func Pow_(x, c *Variable) *Variable {
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

func (p *Pow) Backward(gy ...*Variable) []*Variable {
	x, c := p.input, p.c
	return []*Variable{Mul_(Mul_(c, Pow_(x, Sub_(c, NewVar(1)))), gy[0])}
}

func (p *Pow) String() string {
	return "x ** c"
}

func Sin_(x *Variable) *Variable {
	f := NewFunction(&Sin{input: x})
	return f.forward(x)[0]
}

type Sin struct {
	input *Variable
}

func (s *Sin) Forward(inputs ...*Variable) []*Variable {
	s.input = inputs[0]
	v := NewVar(math.Sin(inputs[0].data))
	out := []*Variable{v}
	return out
}

func (s *Sin) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Mul_(gy[0], Cos_(s.input))}
}

func (s *Sin) String() string {
	return "sin"
}

func Cos_(x *Variable) *Variable {
	f := NewFunction(&Cos{input: x})
	return f.forward(x)[0]
}

type Cos struct {
	input *Variable
}

func (c *Cos) Forward(inputs ...*Variable) []*Variable {
	c.input = inputs[0]
	v := NewVar(math.Cos(inputs[0].data))
	out := []*Variable{v}
	return out
}

func (c *Cos) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Mul_(gy[0], Neg_(Sin_(c.input)))}
}

func (c *Cos) String() string {
	return "cos"
}

func Tanh_(x *Variable) *Variable {
	f := NewFunction(&Tanh{input: x})
	return f.forward(x)[0]
}

type Tanh struct {
	input  *Variable
	output *Variable
}

func (t *Tanh) Forward(inputs ...*Variable) []*Variable {
	t.input = inputs[0]
	v := NewVar(math.Tanh(inputs[0].data))
	out := []*Variable{v}
	t.output = v
	return out
}

func (t *Tanh) Backward(gy ...*Variable) []*Variable {
	y := t.output
	return []*Variable{Mul_(gy[0], Sub_(NewVar(1), Mul_(y, y)))}
}

func (t *Tanh) String() string {
	return "tanh"
}
