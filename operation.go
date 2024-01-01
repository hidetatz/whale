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
func Transpose(v *Variable, axes ...int) (*Variable, error) {
	f := NewFunction(&transpose{axes: axes})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type transpose struct {
	axes []int
}

func (t *transpose) Forward(inputs ...*Variable) ([]*Variable, error) {
	tr, err := inputs[0].data.Transpose(t.axes...)
	if err != nil {
		return nil, err
	}
	return []*Variable{NewVar(tr)}, nil
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
	if slices.Equal(v.data.CopyShape(), shape) {
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
	if ndim != 0 && len(s.axes) != 0 && !s.keepdims {
		sorted := make([]int, len(s.axes))
		copy(sorted, s.axes)
		slices.Sort(sorted)
		for _, a := range sorted {
			// insert a
			shape = append(shape[:a], append([]int{1}, shape[a:]...)...)
		}
	}

	y, err := gy0.data.Reshape(shape...)
	if err != nil {
		return nil, fmt.Errorf("Sum Backward: %w", err)
	}

	gx, err := BroadcastTo(NewVar(y), s.origshape...)
	if err != nil {
		return nil, fmt.Errorf("Sum Backward: %w", err)
	}

	return []*Variable{gx}, nil
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
		return nil, fmt.Errorf("SumTo: %w", err)
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
	x0shape []int
	x1shape []int
}

func (a *add) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Add(inputs[0].data, inputs[1].data)), nil
}

func (a *add) Backward(gy ...*Variable) ([]*Variable, error) {
	gx0, gx1 := gy[0], gy[0]
	if !slices.Equal(a.x0shape, a.x1shape) {
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

func Sub(x0, x1 *Variable) (*Variable, error) {
	f := NewFunction(&sub{x0shape: x0.data.CopyShape(), x1shape: x1.data.CopyShape()})
	y, err := f.forward(x0, x1)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type sub struct {
	x0shape []int
	x1shape []int
}

func (s *sub) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Sub(inputs[0].data, inputs[1].data)), nil
}

func (s *sub) Backward(gy ...*Variable) ([]*Variable, error) {
	gx0 := gy[0]
	gx1, err := Neg(gy[0])
	if err != nil {
		return nil, err
	}

	if !slices.Equal(s.x0shape, s.x1shape) {
		var err error
		gx0, err = SumTo(gx0, s.x0shape...)
		if err != nil {
			return nil, fmt.Errorf("Sub Backward: %w", err)
		}

		gx1, err = SumTo(gx1, s.x1shape...)
		if err != nil {
			return nil, fmt.Errorf("Sub Backward: %w", err)
		}
	}
	return []*Variable{gx0, gx1}, nil
}

func (s *sub) String() string {
	return "-"
}

func Mul(x0, x1 *Variable) (*Variable, error) {
	f := NewFunction(&mul{x0: x0, x1: x1, x0shape: x0.data.CopyShape(), x1shape: x1.data.CopyShape()})
	y, err := f.forward(x0, x1)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type mul struct {
	x0      *Variable
	x1      *Variable
	x0shape []int
	x1shape []int
}

func (m *mul) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Mul(m.x0.data, m.x1.data)), nil
}

func (m *mul) Backward(gy ...*Variable) ([]*Variable, error) {
	gx0, err := Mul(gy[0], m.x1)
	if err != nil {
		return nil, err
	}

	gx1, err := Mul(gy[0], m.x0)
	if err != nil {
		return nil, err
	}

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

func Div(x0, x1 *Variable) (*Variable, error) {
	f := NewFunction(&div{x0: x0, x1: x1, x0shape: x0.data.CopyShape(), x1shape: x1.data.CopyShape()})
	y, err := f.forward(x0, x1)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type div struct {
	x0      *Variable
	x1      *Variable
	x0shape []int
	x1shape []int
}

func (d *div) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Div(inputs[0].data, inputs[1].data)), nil
}

func (d *div) Backward(gy ...*Variable) ([]*Variable, error) {
	gx0, err := Div(gy[0], d.x1)
	if err != nil {
		return nil, err
	}

	t1, err := Neg(d.x0)
	if err != nil {
		return nil, err
	}

	t2, err := Pow(d.x1, NewVar(tensor.FromScalar(2)))
	if err != nil {
		return nil, err
	}

	t3, err := Div(t1, t2)
	if err != nil {
		return nil, err
	}

	gx1, err := Mul(gy[0], t3)
	if err != nil {
		return nil, err
	}

	if !slices.Equal(d.x0shape, d.x1shape) {
		var err error
		gx0, err = SumTo(gx0, d.x0shape...)
		if err != nil {
			return nil, fmt.Errorf("Div Backward: %w", err)
		}
		gx1, err = SumTo(gx1, d.x1shape...)
		if err != nil {
			return nil, fmt.Errorf("Div Backward: %w", err)
		}

	}
	return []*Variable{gx0, gx1}, nil
}

func (d *div) String() string {
	return "/"
}

func Neg(x *Variable) (*Variable, error) {
	f := NewFunction(&neg{})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type neg struct{}

func (n *neg) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Neg(inputs[0].data)), nil
}

func (n *neg) Backward(gy ...*Variable) ([]*Variable, error) {
	gx, err := Neg(gy[0])
	if err != nil {
		return nil, err
	}

	return []*Variable{gx}, nil
}

func (n *neg) String() string {
	return "-x"
}

func Pow(x, c *Variable) (*Variable, error) {
	f := NewFunction(&pow{x: x, c: c})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type pow struct {
	x *Variable
	c *Variable
}

func (p *pow) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Pow(inputs[0].data, p.c.data)), nil
}

func (p *pow) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Sub(p.c, NewVar(tensor.FromScalar(1)))
	if err != nil {
		return nil, fmt.Errorf("Pow Backward: %w", err)
	}

	t2, err := Pow(p.x, t1)
	if err != nil {
		return nil, fmt.Errorf("Pow Backward: %w", err)
	}

	t3, err := Mul(p.c, t2)
	if err != nil {
		return nil, fmt.Errorf("Pow Backward: %w", err)
	}

	gx, err := Mul(t3, gy[0])
	if err != nil {
		return nil, fmt.Errorf("Pow Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (p *pow) String() string {
	return "x^c"
}

func Sin(x *Variable) (*Variable, error) {
	f := NewFunction(&sin{x: x})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}
	return y[0], nil
}

type sin struct {
	x *Variable
}

func (s *sin) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Sin(s.x.data)), nil
}

func (s *sin) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Cos(s.x)
	if err != nil {
		return nil, fmt.Errorf("Sin Backward: %w", err)
	}

	gx, err := Mul(gy[0], t1)
	if err != nil {
		return nil, fmt.Errorf("Sin Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (s *sin) String() string {
	return "sin"
}

func Cos(x *Variable) (*Variable, error) {
	f := NewFunction(&cos{x: x})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type cos struct {
	x *Variable
}

func (c *cos) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Cos(c.x.data)), nil
}

func (c *cos) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Sin(c.x)
	if err != nil {
		return nil, fmt.Errorf("Cos Backward: %w", err)
	}

	t2, err := Neg(t1)
	if err != nil {
		return nil, fmt.Errorf("Cos Backward: %w", err)
	}
	gx, err := Mul(gy[0], t2)
	if err != nil {
		return nil, fmt.Errorf("Cos Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (c *cos) String() string {
	return "cos"
}

func Tanh(x *Variable) (*Variable, error) {
	f := NewFunction(&tanh{x: x})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type tanh struct {
	x *Variable
	y *Variable
}

func (t *tanh) Forward(inputs ...*Variable) ([]*Variable, error) {
	y := NewVar(device.Tanh(t.x.data))
	t.y = y
	return []*Variable{y}, nil
}

func (t *tanh) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Mul(t.y, t.y)
	if err != nil {
		return nil, fmt.Errorf("Tanh Backward: %w", err)
	}

	t2, err := Sub(NewVar(tensor.FromScalar(1)), t1)
	if err != nil {
		return nil, fmt.Errorf("Tanh Backward: %w", err)
	}

	gx, err := Mul(gy[0], t2)
	if err != nil {
		return nil, fmt.Errorf("Tanh Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (t *tanh) String() string {
	return "tanh"
}

func MatMul(x, w *Variable) (*Variable, error) {
	f := NewFunction(&matmul{x: x, w: w})
	y, err := f.forward(x, w)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type matmul struct {
	x *Variable
	w *Variable
}

func (m *matmul) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(device.Dot(m.x.data, m.w.data)), nil
}

func (m *matmul) Backward(gy ...*Variable) ([]*Variable, error) {
	wt, err := Transpose(m.w)
	if err != nil {
		return nil, fmt.Errorf("Matmul Backward: %w", err)
	}

	xt, err := Transpose(m.x)
	if err != nil {
		return nil, fmt.Errorf("Matmul Backward: %w", err)
	}

	gx, err := MatMul(gy[0], wt)
	if err != nil {
		return nil, fmt.Errorf("Matmul Backward: %w", err)
	}

	gw, err := MatMul(xt, gy[0])
	if err != nil {
		return nil, fmt.Errorf("Matmul Backward: %w", err)
	}

	return []*Variable{gx, gw}, nil
}

func (m *matmul) String() string {
	return "matmul"
}
