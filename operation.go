package whale

import (
	"fmt"
	"slices"

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
 * Tensor modification
 */

func Reshape_(v *Variable, shape ...int) *Variable {
	f := NewFunction(&Reshape{shape: shape})
	return f.forward(v)[0]
}

type Reshape struct {
	input     *Variable
	origshape []int
	shape     []int
}

func (r *Reshape) Forward(inputs ...*Variable) []*Variable {
	r.input = inputs[0]
	r.origshape = inputs[0].data.CopyShape()
	y, err := r.input.data.Reshape(r.shape...)
	if err != nil {
		panic(err.Error())
	}
	v := NewVar(y)
	out := []*Variable{v}
	return out
}

func (r *Reshape) Backward(gy ...*Variable) []*Variable {
	return []*Variable{
		Reshape_(gy[0], r.origshape...),
	}
}

func (r *Reshape) String() string {
	return "reshape"
}

func Transpose_(v *Variable) *Variable {
	f := NewFunction(&Transpose{})
	return f.forward(v)[0]
}

type Transpose struct{}

func (t *Transpose) Forward(inputs ...*Variable) []*Variable {
	in := inputs[0]
	return []*Variable{NewVar(in.data.Transpose())}
}

func (t *Transpose) Backward(gy ...*Variable) []*Variable {
	return []*Variable{Transpose_(gy[0])}
}

func (t *Transpose) String() string {
	return "T"
}

func BroadcastTo_(v *Variable, shape ...int) *Variable {
	f := NewFunction(&BroadcastTo{shape: shape})
	return f.forward(v)[0]
}

type BroadcastTo struct {
	origShape []int
	shape     []int
}

func (b *BroadcastTo) Forward(inputs ...*Variable) []*Variable {
	in := inputs[0]
	b.origShape = in.data.CopyShape()
	y, err := in.data.BroadcastTo(b.shape...)
	if err != nil {
		panic(err)
	}

	return []*Variable{NewVar(y)}
}

func (b *BroadcastTo) Backward(gy ...*Variable) []*Variable {
	return []*Variable{SumTo_(gy[0], b.origShape...)}
}

func (b *BroadcastTo) String() string {
	return "BroadcastTo"
}

/*
 * Sum
 */

func Sum_(v *Variable, keepdims bool, axes ...int) *Variable {
	f := NewFunction(&Sum{keepdims: keepdims, axes: axes})
	return f.forward(v)[0]
}

type Sum struct {
	input     *Variable
	keepdims  bool
	axes      []int
	origshape []int
}

func (s *Sum) Forward(inputs ...*Variable) []*Variable {
	s.input = inputs[0]
	s.origshape = s.input.data.CopyShape()
	y, err := s.input.data.Sum(s.keepdims, s.axes...)
	if err != nil {
		panic(err.Error())
	}
	v := NewVar(y)
	out := []*Variable{v}
	return out
}

func (s *Sum) Backward(gy ...*Variable) []*Variable {
	gy0 := gy[0]
	ndim := len(s.origshape)

	shape := gy0.data.CopyShape()
	if !(ndim == 0 || len(s.axes) == 0 || s.keepdims) {
		actualAxes := make([]int, len(s.axes))
		for i, axis := range s.axes {
			if axis >= 0 {
				actualAxes[i] = s.axes[i]
			} else {
				actualAxes[i] = s.axes[i] + ndim
			}
		}
		shape = gy0.data.CopyShape()
		slices.Sort(actualAxes)
		for _, a := range actualAxes {
			// insert a
			shape = append(shape[:1], append([]int{a}, shape[1:]...)...)
		}
	}

	y, err := gy0.data.Reshape(shape...)
	if err != nil {
		panic(err.Error())
	}
	return []*Variable{NewVar(y)}
}

func (s *Sum) String() string {
	return "sum"
}

func SumTo_(v *Variable, shape ...int) *Variable {
	f := NewFunction(&SumTo{shape: shape})
	return f.forward(v)[0]
}

type SumTo struct {
	input     *Variable
	shape     []int
	origshape []int
}

func (s *SumTo) Forward(inputs ...*Variable) []*Variable {
	s.input = inputs[0]
	s.origshape = s.input.data.CopyShape()
	y, err := s.input.data.SumTo(s.shape...)
	if err != nil {
		panic(err.Error())
	}
	v := NewVar(y)
	out := []*Variable{v}
	return out
}

func (s *SumTo) Backward(gy ...*Variable) []*Variable {
	y, err := gy[0].data.BroadcastTo(s.origshape...)
	if err != nil {
		panic(err.Error())
	}
	return []*Variable{NewVar(y)}
}

func (s *SumTo) String() string {
	return "sumto"
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
	y := tensor.All(2, s.input.data.Shape()...)
	v := NewVar(device.Pow(s.input.data, y))
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

func sameSlice(x1, x2 []int) bool {
	if len(x1) != len(x2) {
		return false
	}

	for i := range x1 {
		if x1[i] != x2[i] {
			return false
		}
	}

	return true
}

func Add_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Add{})
	return f.forward(x1, x2)[0]
}

type Add struct {
	inputs  []*Variable
	x0Shape []int
	x1Shape []int
}

func (a *Add) Forward(inputs ...*Variable) []*Variable {
	a.inputs = inputs
	a.x0Shape = inputs[0].data.CopyShape()
	a.x1Shape = inputs[1].data.CopyShape()
	v := NewVar(device.Add(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (a *Add) Backward(gy ...*Variable) []*Variable {
	gx0, gx1 := gy[0], gy[0]
	if !sameSlice(a.x0Shape, a.x1Shape) {
		gx0 = SumTo_(gx0, a.x0Shape...)
		gx1 = SumTo_(gx1, a.x1Shape...)
	}

	return []*Variable{gx0, gx1}
}

func (a *Add) String() string {
	return "+"
}

func Sub_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Sub{})
	return f.forward(x1, x2)[0]
}

type Sub struct {
	inputs  []*Variable
	x0Shape []int
	x1Shape []int
}

func (s *Sub) Forward(inputs ...*Variable) []*Variable {
	s.inputs = inputs
	s.x0Shape = inputs[0].data.CopyShape()
	s.x1Shape = inputs[1].data.CopyShape()
	v := NewVar(device.Sub(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (s *Sub) Backward(gy ...*Variable) []*Variable {
	gx0, gx1 := gy[0], Neg_(gy[0])
	if !sameSlice(s.x0Shape, s.x1Shape) {
		gx0 = SumTo_(gx0, s.x0Shape...)
		gx1 = SumTo_(gx1, s.x1Shape...)
	}
	return []*Variable{gx0, gx1}
}

func (s *Sub) String() string {
	return "-"
}

func Mul_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Mul{})
	return f.forward(x1, x2)[0]
}

type Mul struct {
	inputs  []*Variable
	x0Shape []int
	x1Shape []int
}

func (m *Mul) Forward(inputs ...*Variable) []*Variable {
	m.inputs = inputs
	m.x0Shape = inputs[0].data.CopyShape()
	m.x1Shape = inputs[1].data.CopyShape()
	v := NewVar(device.Mul(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (m *Mul) Backward(gy ...*Variable) []*Variable {
	x0, x1 := m.inputs[0], m.inputs[1]
	gx0, gx1 := Mul_(gy[0], x1), Mul_(gy[0], x0)
	if !sameSlice(m.x0Shape, m.x1Shape) {
		gx0 = SumTo_(gx0, m.x0Shape...)
		gx1 = SumTo_(gx1, m.x1Shape...)
	}
	return []*Variable{gx0, gx1}
}

func (m *Mul) String() string {
	return "*"
}

func Div_(x1, x2 *Variable) *Variable {
	f := NewFunction(&Div{})
	return f.forward(x1, x2)[0]
}

type Div struct {
	inputs  []*Variable
	x0Shape []int
	x1Shape []int
}

func (d *Div) Forward(inputs ...*Variable) []*Variable {
	d.inputs = inputs
	d.x0Shape = inputs[0].data.CopyShape()
	d.x1Shape = inputs[1].data.CopyShape()
	v := NewVar(device.Div(inputs[0].data, inputs[1].data))
	out := []*Variable{v}
	return out
}

func (d *Div) Backward(gy ...*Variable) []*Variable {
	x0, x1 := d.inputs[0], d.inputs[1]
	gx0, gx1 := Div_(gy[0], x1), Mul_(gy[0], Div_(Neg_(x0), Pow_(x1, NewVar(tensor.FromScalar(2)))))
	if !sameSlice(d.x0Shape, d.x1Shape) {
		gx0 = SumTo_(gx0, d.x0Shape...)
		gx1 = SumTo_(gx1, d.x1Shape...)
	}
	return []*Variable{gx0, gx1}
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
	v := NewVar(device.Neg(inputs[0].data))
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
	v := NewVar(device.Pow(inputs[0].data, p.c.data))
	out := []*Variable{v}
	return out
}

func (p *Pow) Backward(gy ...*Variable) []*Variable {
	x, c := p.input, p.c
	return []*Variable{Mul_(Mul_(c, Pow_(x, Sub_(c, NewVar(tensor.FromScalar(1))))), gy[0])}
}

func (p *Pow) String() string {
	return "x^c"
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
	v := NewVar(device.Sin(inputs[0].data))
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
	v := NewVar(device.Cos(inputs[0].data))
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
	v := NewVar(device.Tanh(inputs[0].data))
	out := []*Variable{v}
	t.output = v
	return out
}

func (t *Tanh) Backward(gy ...*Variable) []*Variable {
	y := t.output
	return []*Variable{Mul_(gy[0], Sub_(NewVar(tensor.FromScalar(1)), Mul_(y, y)))}
}

func (t *Tanh) String() string {
	return "tanh"
}

func MatMul_(x, w *Variable) *Variable {
	f := NewFunction(&MatMul{w: w})
	return f.forward(x)[0]
}

type MatMul struct {
	x *Variable
	w *Variable
}

func (m *MatMul) Forward(inputs ...*Variable) []*Variable {
	m.x = inputs[0]
	v := NewVar(device.MatMul(m.x.data, m.w.data))
	out := []*Variable{v}
	return out
}

func (m *MatMul) Backward(gy ...*Variable) []*Variable {
	y1, y2 := gy[0], gy[0]
	gx := MatMul_(y1, m.w.Transpose())
	gw := MatMul_(m.x.Transpose(), y2)
	return []*Variable{gx, gw}
}

func (m *MatMul) String() string {
	return "matmul"
}
