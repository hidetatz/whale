package main

import (
	"fmt"
	"iter"
	"os"
	"slices"
	"strings"
)

var debug bool

func initDebug() {
	debug = os.Getenv("DEBUG") == "1"
}

func init() {
	initDebug()
	initRunner()
}

type dimension struct {
	shape   []int
	strides []int
	offset  int
}

type Tensor struct {
	data         []float32
	dim          *dimension
	materialized bool

	// calculation graph
	op      op
	creator *calc
	ctx     *context
	inputs  []*Tensor

	// gradient
	grad *Tensor
}

func (t *Tensor) String() string {
	return fmt.Sprintf("{\n%s}", t.string(1))
}

func (t *Tensor) string(depth int) string {
	sb := strings.Builder{}
	indent := strings.Repeat("  ", depth)

	writeraw := func(format string, a ...any) { sb.WriteString(fmt.Sprintf(format, a...)) }
	write := func(format string, a ...any) { writeraw(indent+format+"\n", a...) }

	switch t.op {
	case ops.constant:
		write("op: const,")
		write("data: %v", t.data)
	case ops.add, ops.mul:
		if t.op == ops.add {
			write("op: +,")
		} else if t.op == ops.mul {
			write("op: *,")
		}
		write("left: {")
		writeraw(t.inputs[0].string(depth + 1))
		write("}")
		write("right: {")
		writeraw(t.inputs[1].string(depth + 1))
		write("},")
	case ops.expand:
		write("op: expand,")
		write("from: {")
		writeraw(t.inputs[0].string(depth + 1))
		write("}")
		write("to_shape: %v,", t.dim.shape)
	default:
		panic("switch-case is not exhaustive!")
	}

	return sb.String()
}

func newdimension(shape ...int) *dimension {
	product := func(arr []int) int {
		p := 1
		for i := range arr {
			p *= arr[i]
		}
		return p
	}

	strides := make([]int, len(shape))
	for i := range len(shape) {
		strides[i] = product(shape[i+1:])
	}

	return &dimension{
		shape:   shape,
		strides: strides,
		offset:  0,
	}
}

// returns broadcasted shape and true if broadcastable.
func (d *dimension) broadcastable(d2 *dimension) ([]int, bool) {
	if slices.Equal(d.shape, d2.shape) {
		return d.shape, true
	}

	tmpd := slices.Clone(d.shape)
	tmpd2 := slices.Clone(d2.shape)

	// prepend 1s to be the same length
	if len(tmpd) > len(tmpd2) {
		tmpd2 = slices.Concat(slices.Repeat([]int{1}, len(tmpd)-len(tmpd2)), tmpd2)
	} else if len(tmpd2) > len(tmpd) {
		tmpd = slices.Concat(slices.Repeat([]int{1}, len(tmpd2)-len(tmpd)), tmpd)
	}

	broadcastedshape := []int{}
	for s1, s2 := range zip(tmpd, tmpd2) {
		if s1 != s2 && s1 != 1 && s2 != 1 {
			return nil, false
		}
		broadcastedshape = append(broadcastedshape, max(s1, s2))
	}

	return broadcastedshape, true
}

/*******************************
 *
 * Tensor factory function
 *
 *******************************/

func Scalar(data float32) *Tensor {
	return &Tensor{data: []float32{data}, dim: newdimension(), materialized: true}
}

func Vector(data []float32) *Tensor {
	return &Tensor{data: data, dim: newdimension(len(data)), materialized: true}
}

func newFromCalc(calc *calc, inputs ...*Tensor) *Tensor {
	ctx := newcontext()
	return newFromCalcWithCtx(calc, ctx, inputs...)
}

func newFromCalcWithCtx(calc *calc, ctx *context, inputs ...*Tensor) *Tensor {
	ctx.inputs = inputs
	t := calc.do(ctx, inputs...)
	t.creator = calc
	t.ctx = ctx
	return t
}

/*******************************
 *
 * Calculation
 *
 *******************************/

/*
 * Arithmetic
 */

// func (t *Tensor) Recip() *Tensor {
// 	return fromfunc(&recip{}, t)
// }

// func (t *Tensor) Neg() *Tensor {
// 	return t.Mul(Vector([]float32{-1}))
// }

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return newFromCalc(calculations.add, t.broadcasted(t2)...)
}

// func (t *Tensor) Sub(t2 *Tensor) *Tensor {
// 	return t.Add(t2.Neg())
// }

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	bt := t.broadcasted(t2)
	fmt.Println(11111, bt[0].inputs[0])
	fmt.Println(22222, bt[1].inputs[0])
	return newFromCalc(calculations.mul, bt...)
}

// func (t *Tensor) Div(t2 *Tensor) *Tensor {
// 	return t.Mul(t2.Recip())
// }

/*******************************
 *
 * Transformation
 *
 *******************************/

func (t *Tensor) broadcasted(t2 *Tensor) []*Tensor {
	// if slices.Equal(t.dim.shape, t2.dim.shape) {
	// 	return []*Tensor{t, t2}
	// }

	shape, ok := t.dim.broadcastable(t2.dim)
	if !ok {
		panic(fmt.Sprintf("broadcast is impossible on shape %d and %d", t.dim.shape, t2.dim.shape))
	}

	return []*Tensor{
		newFromCalcWithCtx(calculations.expand, newcontext().setshape(shape...), t),
		newFromCalcWithCtx(calculations.expand, newcontext().setshape(shape...), t2),
	}
}

/*******************************
 *
 * Gradients
 *
 *******************************/

func (t *Tensor) Backprop() {
	if t.grad == nil {
		t.grad = Vector([]float32{1})
	}

	flatten := func(t *Tensor) []*Tensor {
		visited := make(map[*Tensor]bool)
		var tensors []*Tensor
		var dfs func(*Tensor)
		dfs = func(_t *Tensor) {
			if _t.creator == nil {
				return
			}

			if visited[_t] {
				return
			}
			visited[_t] = true

			tensors = append(tensors, _t)
			for _, input := range _t.inputs {
				dfs(input)
			}
		}

		dfs(t)
		return tensors
	}

	for _, tensor := range flatten(t) {
		grads := tensor.creator.differential(tensor.ctx, tensor.grad)

		for input, grad := range zip(tensor.inputs, grads) {
			if input.grad == nil {
				input.grad = grad
			} else {
				y := input.grad.Add(grad)
				input.grad = y
			}
		}
	}
}

func main() {
	t := Vector([]float32{1, 2})
	t2 := Vector([]float32{3})
	t3 := t.Mul(t2)

	fmt.Println(t3.Materialize())

	// fmt.Println(t3.grad, t2.grad, t.grad)
	// t3.Backprop()
	// fmt.Println(t3.grad.Materialize())
	// fmt.Println(t2.grad.Materialize())
	// fmt.Println(t.grad.Materialize())
}

// zip utility function
func zip[K, V any](ks []K, vs []V) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for i := range min(len(ks), len(vs)) {
			yield(ks[i], vs[i])
		}
	}
}
