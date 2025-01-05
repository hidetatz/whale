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
	initBackend()
}

/*
 * dimension
 */

type dimension struct {
	shape   []int
	strides []int
	offset  int
}

func newdim(shape ...int) *dimension {
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

func (d *dimension) String() string {
	return fmt.Sprintf("{shape: %v, strides: %v, offset: %v}", d.shape, d.strides, d.offset)
}

func (d *dimension) copy() *dimension {
	return &dimension{
		shape:   slices.Clone(d.shape),
		strides: slices.Clone(d.strides),
		offset:  d.offset,
	}
}

func (d *dimension) ndim() int {
	return len(d.shape)
}

func (d *dimension) size() int {
	return product(d.shape)
}

func (d *dimension) expand(newshape ...int) *dimension {
	if slices.Equal(d.shape, newshape) {
		return d
	}

	if len(d.shape) > len(newshape) {
		panic("cannot expand to smaller dimensions")
	}

	delta := len(newshape) - len(d.shape)
	newstrides := make([]int, len(newshape))

	for i := d.ndim() - 1; 0 <= i; i-- {
		if newshape[delta+i] == d.shape[i] && d.shape[i] == 1 {
			newstrides[delta+i] = 0
			continue
		}

		if newshape[delta+i] == d.shape[i] && d.shape[i] != 1 {
			newstrides[delta+i] = d.strides[i]
			continue
		}

		if d.shape[i] != 1 {
			panic("expand: dimension number is not 1")
		}

		newstrides[delta+i] = 0
	}

	return &dimension{
		shape:   newshape,
		strides: newstrides,
		offset:  d.offset,
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

/*
 * Tensor
 */

type Tensor struct {
	data         []float32
	dim          *dimension
	materialized bool

	// calculation graph
	op      operation
	creator calculation
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
	case op_const:
		write("op: const,")
		write("dim: %v,", t.dim)
		write("data: %v", t.data)

	case op_recip:
		write("dim: %v,", t.dim)
		write("op: %v,", t.op)
		write("input: {")
		writeraw(t.inputs[0].string(depth + 1))
		write("}")

	case op_add, op_mul:
		write("dim: %v,", t.dim)
		write("op: %v,", t.op)
		write("left: {")
		writeraw(t.inputs[0].string(depth + 1))
		write("}")
		write("right: {")
		writeraw(t.inputs[1].string(depth + 1))
		write("},")

	case op_expand:
		write("op: expand,")
		write("dim: %v,", t.dim)
		write("from: {")
		writeraw(t.inputs[0].string(depth + 1))
		write("}")

	default:
		panic("switch-case is not exhaustive!")
	}

	return sb.String()
}

/*******************************
 *
 * Tensor factory function
 *
 *******************************/

func newconst(data []float32, shape ...int) *Tensor {
	return &Tensor{op: op_const, data: data, dim: newdim(shape...), materialized: true}
}

func Scalar(data float32) *Tensor {
	return newconst([]float32{data})
}

func Vector(data []float32) *Tensor {
	return newconst(data, len(data))
}

func newFromCalc(calc calculation, inputs ...*Tensor) *Tensor {
	t := calc.do(inputs...)
	t.creator = calc
	return t
}

func (t *Tensor) IsScalar() bool {
	return len(t.dim.shape) == 0
}

func (t *Tensor) IsVector() bool {
	return len(t.dim.shape) == 1
}

func (t *Tensor) Size() int {
	return t.dim.size()
}

func product(arr []int) int {
	p := 1
	for i := range arr {
		p *= arr[i]
	}
	return p
}

/*******************************
 *
 * Calculation
 *
 *******************************/

/*
 * Arithmetic
 */

func (t *Tensor) Recip() *Tensor {
	return newFromCalc(&calcRecip{}, t)
}

// func (t *Tensor) Neg() *Tensor {
// 	return t.Mul(Vector([]float32{-1}))
// }

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return newFromCalc(&calcAdd{}, t.broadcasted(t2)...)
}

// func (t *Tensor) Sub(t2 *Tensor) *Tensor {
// 	return t.Add(t2.Neg())
// }

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return newFromCalc(&calcMul{}, t.broadcasted(t2)...)
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
	if slices.Equal(t.dim.shape, t2.dim.shape) {
		return []*Tensor{t, t2}
	}

	shape, ok := t.dim.broadcastable(t2.dim)
	if !ok {
		panic(fmt.Sprintf("broadcast is impossible on shape %d and %d", t.dim.shape, t2.dim.shape))
	}

	expandedT := t
	expandedT2 := t2

	if !slices.Equal(t.dim.shape, shape) {
		expandedT = newFromCalc(&calcExpand{shape: shape}, t)
	}

	if !slices.Equal(t2.dim.shape, shape) {
		expandedT2 = newFromCalc(&calcExpand{shape: shape}, t2)
	}

	return []*Tensor{expandedT, expandedT2}
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
		grads := tensor.creator.differential(tensor.grad)

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

/*******************************
 *
 * Materialization
 *
 *******************************/

func (t *Tensor) Materialize() []float32 {
	result, err := compute(t)
	if err != nil {
		panic(err)
	}

	t.data = result
	return t.data
}

func main() {
	t := Vector([]float32{2})
	t2 := Vector([]float32{3, 5})
	t3 := t.Mul(t2)
	t4 := t3.Recip()
	t5 := t4.Add(Vector([]float32{100, 200}))
	fmt.Println(t5.Materialize())

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
