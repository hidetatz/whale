package whale

import (
	"fmt"
	"slices"

	"github.com/hidetatz/whale/tensor"
)

func copySlice(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
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
	if slices.Equal(v.data.Shape, shape) {
		return v, nil
	}

	f := NewFunction(&reshape{origshape: v.data.Shape, shape: shape})
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
	y := inputs[0].data.Reshape(r.shape...)
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
	tr := inputs[0].data.Transpose(t.axes...)
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
	if slices.Equal(v.data.Shape, shape) {
		return v, nil
	}

	f := NewFunction(&broadcastTo{origshape: v.data.Shape, shape: shape})
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
	y := inputs[0].data.BroadcastTo(b.shape...)
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
	f := NewFunction(&sum{keepdims: keepdims, axes: axes, origshape: v.data.Shape})
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
	y := inputs[0].data.Sum(s.keepdims, s.axes...)
	return asvars(y), nil
}

func (s *sum) Backward(gy ...*Variable) ([]*Variable, error) {
	gy0 := gy[0]
	ndim := len(s.origshape)

	shape := copySlice(gy0.data.Shape)
	if ndim != 0 && len(s.axes) != 0 && !s.keepdims {
		axes := copySlice(s.axes)
		slices.Sort(axes)
		for _, a := range axes {
			// insert a
			shape = append(shape[:a], append([]int{1}, shape[a:]...)...)
		}
	}

	y := gy0.data.Reshape(shape...)
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
	if slices.Equal(v.data.Shape, shape) {
		return v, nil
	}

	f := NewFunction(&sumTo{origshape: v.data.Shape, shape: shape})
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
	y := inputs[0].data.SumTo(s.shape...)
	return asvars(y), nil
}

func (s *sumTo) Backward(gy ...*Variable) ([]*Variable, error) {
	y := gy[0].data.BroadcastTo(s.origshape...)
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
	v := NewVar(inputs[0].data.Exp())
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
	f := NewFunction(&add{x0shape: x0.data.Shape, x1shape: x1.data.Shape})
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
	add := inputs[0].data.Add(inputs[1].data)
	return asvars(add), nil
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
	f := NewFunction(&sub{x0shape: x0.data.Shape, x1shape: x1.data.Shape})
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
	sub := inputs[0].data.Sub(inputs[1].data)
	return asvars(sub), nil
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
	f := NewFunction(&mul{x0: x0, x1: x1, x0shape: x0.data.Shape, x1shape: x1.data.Shape})
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
	mul := m.x0.data.Mul(m.x1.data)
	return asvars(mul), nil
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
	f := NewFunction(&div{x0: x0, x1: x1, x0shape: x0.data.Shape, x1shape: x1.data.Shape})
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
	div := inputs[0].data.Div(inputs[1].data)
	return asvars(div), nil
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

	t2, err := Pow(d.x1, NewVar(tensor.Scalar(2)))
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
	return asvars(inputs[0].data.Neg()), nil
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
	pow := inputs[0].data.Pow(p.c.data)
	return asvars(pow), nil
}

func (p *pow) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Sub(p.c, NewVar(tensor.Scalar(1)))
	if err != nil {
		return nil, fmt.Errorf("Pow Backward (Sub): %w", err)
	}

	t2, err := Pow(p.x, t1)
	if err != nil {
		return nil, fmt.Errorf("Pow Backward (Pow): %w", err)
	}

	t3, err := Mul(p.c, t2)
	if err != nil {
		return nil, fmt.Errorf("Pow Backward (Mul1): %w", err)
	}

	gx, err := Mul(t3, gy[0])
	if err != nil {
		return nil, fmt.Errorf("Pow Backward (Mul2): %w", err)
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
	return asvars(s.x.data.Sin()), nil
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
	return asvars(c.x.data.Cos()), nil
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
	y := NewVar(t.x.data.Tanh())
	t.y = y
	return []*Variable{y}, nil
}

func (t *tanh) Backward(gy ...*Variable) ([]*Variable, error) {
	t1, err := Mul(t.y, t.y)
	if err != nil {
		return nil, fmt.Errorf("Tanh Backward: %w", err)
	}

	t2, err := Sub(NewVar(tensor.Scalar(1)), t1)
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
	return asvars(m.x.data.Matmul(m.w.data)), nil
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

func Clip(x *Variable, min, max float32) (*Variable, error) {
	f := NewFunction(&clip{x: x, min: min, max: max})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type clip struct {
	x        *Variable
	min, max float32
}

func (c *clip) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(c.x.data.Clip(c.min, c.max)), nil
}

func (c *clip) Backward(gy ...*Variable) ([]*Variable, error) {
	minMask := c.x.data.Bool(func(f float32) bool {
		return f >= c.min
	})

	maxMask := c.x.data.Bool(func(f float32) bool {
		return f <= c.max
	})

	mask, err := Mul(NewVar(minMask), NewVar(maxMask))
	if err != nil {
		return nil, fmt.Errorf("Clip Backward: %w", err)
	}

	gx, err := Mul(gy[0], mask)
	if err != nil {
		return nil, fmt.Errorf("Clip Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (c *clip) String() string {
	return "clip"
}

func Log(x *Variable) (*Variable, error) {
	f := NewFunction(&log{x: x})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type log struct {
	x *Variable
}

func (l *log) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(l.x.data.Log()), nil
}

func (l *log) Backward(gy ...*Variable) ([]*Variable, error) {
	gx, err := Div(gy[0], l.x)
	if err != nil {
		return nil, fmt.Errorf("Log Backward: %w", err)
	}

	return []*Variable{gx}, nil
}

func (l *log) String() string {
	return "log"
}

func Index(x *Variable, indices ...*Variable) (*Variable, error) {
	f := NewFunction(&index{x: x, indices: indices})
	y, err := f.forward(x)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type index struct {
	x       *Variable
	indices []*Variable
}

func (i *index) Forward(inputs ...*Variable) ([]*Variable, error) {
	args := make([]*tensor.IndexArg, len(i.indices))
	for i, idx := range i.indices {
		args[i] = tensor.List(idx.data)
	}
	t := i.x.data.Index(args...)
	return asvars(t), nil
}

func (i *index) Backward(gy ...*Variable) ([]*Variable, error) {
	v, err := indexGrad(i.x.data.Shape, i.indices, gy[0])
	if err != nil {
		return nil, err
	}

	return []*Variable{v}, nil
}

func (i *index) String() string {
	return "index"
}

func indexGrad(xshape []int, indices []*Variable, gy *Variable) (*Variable, error) {
	f := NewFunction(&indexgrad{indices: indices, xshape: xshape})
	y, err := f.forward(gy)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type indexgrad struct {
	xshape  []int
	indices []*Variable
}

func (i *indexgrad) Forward(inputs ...*Variable) ([]*Variable, error) {
	gx := tensor.Zeros(i.xshape...)

	in := make([]*tensor.IndexArg, len(i.indices))
	for i, index := range i.indices {
		in[i] = tensor.List(index.data)
	}

	err := tensor.ADD.At(gx, in, inputs[0].data)
	if err != nil {
		return nil, err
	}

	return asvars(gx), nil
}

func (i *indexgrad) Backward(gy ...*Variable) ([]*Variable, error) {
	v, err := Index(gy[0], i.indices...)
	if err != nil {
		return nil, err
	}

	return []*Variable{v}, nil
}

func (i *indexgrad) String() string {
	return "indexGrad"
}
