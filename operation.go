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

func sameSlice(x1, x2 []int) bool {
	return slices.Equal(x1, x2)
}

func asvars(t *tensor.Tensor) []*Variable {
	return []*Variable{NewVar(t)}
}

// Op is an arbitrary operation which accepts tensors as arguments,
// and returns computed tensors.
type Op interface {
	fmt.Stringer

	// Forward computes tensors.
	Forward(inputs ...*Variable) ([]*Variable, error)

	// Backward computes the derivative for the operation.
	Backward(grads ...*Variable) ([]*Variable, error)
}

/*
 * Tensor modification
 */

// Reshape reshapes the given tensor to the specified shape.
func Reshape(v *Variable, shape ...int) (*Variable, error) {
	if slices.Equal(v.data.Shape(), shape) {
		return v, nil
	}

	f := NewFunction(&reshape{origshape: v.data.CopyShape(), shape: shape})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type reshape struct {
	origshape []int
	shape     []int
}

func (r *reshape) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.Reshape(r.shape...)
	if err != nil {
		return nil, fmt.Errorf("reshape: %w", err)
	}

	return asvars(y), nil
}

func (r *reshape) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := Reshape(gy[0], r.origshape...)
	if err != nil {
		return nil, fmt.Errorf("reshape backward: %w", err)
	}
	return []*Variable{y}, nil
}

func (r *reshape) String() string { return "reshape" }

// Transpose transposes the tensor.
func Transpose(v *Variable) (*Variable, error) {
	f := NewFunction(&transpose{})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type transpose struct{}

func (t *transpose) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(inputs[0].data.Transpose()), nil
}

func (t *transpose) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := Transpose(gy[0])
	if err != nil {
		return nil, err
	}
	return []*Variable{y}, nil
}

func (t *transpose) String() string {
	return "T"
}

// BroadcastTo broadcasts the tensor to the given shape.
func BroadcastTo(v *Variable, shape ...int) (*Variable, error) {
	if sameSlice(v.data.CopyShape(), shape) {
		return v, nil
	}

	f := NewFunction(&broadcastTo{origshape: v.data.CopyShape(), shape: shape})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type broadcastTo struct {
	origshape []int
	shape     []int
}

func (b *broadcastTo) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.BroadcastTo(b.shape...)
	if err != nil {
		return nil, fmt.Errorf("BroadcastTo: %w", err)
	}

	return asvars(y), nil
}

func (b *broadcastTo) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := SumTo(gy[0], b.origshape...)
	if err != nil {
		return nil, fmt.Errorf("BroadcastTo Backward: %w", err)
	}
	return []*Variable{y}, nil
}

func (b *broadcastTo) String() string {
	return "BroadcastTo"
}

/*
 * Sum
 */

func Sum(v *Variable, keepdims bool, axes ...int) (*Variable, error) {
	f := NewFunction(&sum{keepdims: keepdims, axes: axes, origshape: v.data.CopyShape()})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type sum struct {
	keepdims  bool
	axes      []int
	origshape []int
}

func (s *sum) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.Sum(s.keepdims, s.axes...)
	if err != nil {
		return nil, fmt.Errorf("Sum: %w", err)
	}

	return asvars(y), nil
}

func (s *sum) Backward(gy ...*Variable) ([]*Variable, error) {
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
		return nil, fmt.Errorf("Sum Backward: %w", err)
	}

	return asvars(y), nil
}

func (s *sum) String() string {
	return "sum"
}

func SumTo(v *Variable, shape ...int) (*Variable, error) {
	if slices.Equal(v.data.CopyShape(), shape) {
		return v, nil
	}

	f := NewFunction(&sumTo{origshape: v.data.CopyShape(), shape: shape})
	y, err := f.forward(v)

	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type sumTo struct {
	shape     []int
	origshape []int
}

func (s *sumTo) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.SumTo(s.shape...)
	if err != nil {
		return nil, fmt.Errorf("SumTo: %w")
	}

	return asvars(y), nil
}

func (s *sumTo) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := gy[0].data.BroadcastTo(s.origshape...)
	if err != nil {
		return nil, fmt.Errorf("SumTo Backward: %w", err)
	}
	return asvars(y), nil
}

func (s *sumTo) String() string {
	return "sumto"
}

func Square(v *Variable) (*Variable, error) {
	f := NewFunction(&square{input: v})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type square struct {
	input *Variable
}

func (s *square) Forward(inputs ...*Variable) ([]*Variable, error) {
	// y := tensor.All(2, inputs[0].data.CopyShape()...)
	return asvars(device.Pow(s.input.data, tensor.FromScalar(2))), nil
}

func (s *square) Backward(gy ...*Variable) ([]*Variable, error) {
	m1, err := Mul(NewVar(tensor.FromScalar(2)), s.input)
	if err != nil {
		return nil, fmt.Errorf("Square Backward: %w", err)
	}

	m2, err := Mul(m1, gy[0])
	if err != nil {
		return nil, fmt.Errorf("Square Backward: %w", err)
	}

	return []*Variable{m2}, nil
}

func (s *square) String() string {
	return "^2"
}

/*
 * Exp
 */

func Exp(v *Variable) (*Variable, error) {
	f := NewFunction(&exp{})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type exp struct {
	// einput  *Variable
	output *Variable
}

func (e *exp) Forward(inputs ...*Variable) ([]*Variable, error) {
	v := NewVar(device.Exp(inputs[0].data))
	e.output = v
	return []*Variable{v}, nil
}

func (e *exp) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := Mul(e.output, gy[0])
	if err != nil {
		return nil, fmt.Errorf("Exp Backward: %w", err)
	}

	return []*Variable{y}, nil
}

func (e *exp) String() string {
	return "e^x"
}

/*
 * Add, Sub, Mul, Div, Pow
 */

func Add(x0, x1 *Variable) (*Variable, error) {
	f := NewFunction(&add{x0shape: x0.data.CopyShape(), x1shape: x1.data.CopyShape()})
	y, err := f.forward(x0, x1)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type add struct {
	// , niolx0      *Variable
	// , niolx1      *Variable
	x0shape []int
	x1shape []int
}

func (a *add) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Add(inputs[0].data, inputs[1].data)), nil
}

func (a *add) Backward(gy ...*Variable) ([]*Variable, error) {
	gx0, gx1 := gy[0], gy[0]
	if !sameSlice(a.x0shape, a.x1shape) {
		var err error
		gx0, err = SumTo(gx0, a.x0shape...)
		if err != nil {
			return nil, fmt.Errorf("Add Backward: %w", err)
		}
		gx1, err = SumTo(gx1, a.x1shape...)
		if err != nil {
			return nil, fmt.Errorf("Add Backward: %w", err)
		}
	}

	return []*Variable{gx0, gx1}, nil
}

func (a *add) String() string {
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

func Mul(x0, x1 *Variable) (*Variable, error) {
	f := NewFunction(&mul{x0: x0, x1: x1, x0shape: x0.CopyShape(), x1shape: x1.CopyShape()})
	y, err := f.forward(x1, x2)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type mul struct {
	x0  *Variable
	x1  *Variable
	x0shape []int
	x1shape []int
}

func (m *mul) Forward(inputs ...*Variable) []*Variable {
	return asvars(NewVar(device.Mul(m.x0.data, m.x1.data))), nil
}

func (m *mul) Backward(gy ...*Variable) []*Variable {
	gx0, gx1 := Mul(gy[0], m.x1), Mul(gy[0], m.x0)

	if !slices.Equal(m.x0shape, m.x1shape) {
		var err error
		gx0, err = SumTo(gx0, m.x0shape...)
		if err != nil {
			return nil, fmt.Errorf("Mul Backward: %w", err)
		}
		gx1, err = SumTo(gx1, m.x1shape...)
		if err != nil {
			return nil, fmt.Errorf("Mul Backward: %w", err)
		}

	}
	return []*Variable{gx0, gx1}, nil
}

func (m *mul) String() string {
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
	f := NewFunction(&MatMul{})
	return f.forward(x, w)[0]
}

type MatMul struct {
	x *Variable
	w *Variable
}

func (m *MatMul) Forward(inputs ...*Variable) []*Variable {
	m.x = inputs[0]
	m.w = inputs[1]
	y := device.Dot(m.x.data, m.w.data)
	v := NewVar(y)
	out := []*Variable{v}
	return out
}

func (m *MatMul) Backward(gy ...*Variable) []*Variable {
	wt := m.w.Transpose()
	xt := m.x.Transpose()
	gx := MatMul_(gy[0], wt)
	gw := MatMul_(xt, gy[0])
	return []*Variable{gx, gw}
}

func (m *MatMul) String() string {
	return "matmul"
}
