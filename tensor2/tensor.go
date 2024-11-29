package main

import (
	"fmt"
	"os"
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
	inputs []*Tensor
	op     op
	dim    *dimension

	data         []float32
	materialized bool

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
	}

	return sb.String()
}

func newDimension(shape ...int) *dimension {
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
// func (d *dimension) broadcastable(d2 *dimension) ([]int, bool) {

// }

/*******************************
 *
 * Tensor factory function
 *
 *******************************/

func Scalar(data float32) *Tensor {
	return &Tensor{data: []float32{data}, dim: newDimension(), materialized: true}
}

func Vector(data []float32) *Tensor {
	return &Tensor{data: data, dim: newDimension(len(data)), materialized: true}
}

func fromcalculation(c *calculation, inputs ...*Tensor) *Tensor {
	return c.do(inputs...)
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
	return fromcalculation(calculations.add, t, t2)
}

// func (t *Tensor) Sub(t2 *Tensor) *Tensor {
// 	return t.Add(t2.Neg())
// }

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return fromcalculation(calculations.mul, t, t2)
}

// func (t *Tensor) Div(t2 *Tensor) *Tensor {
// 	return t.Mul(t2.Recip())
// }

/*******************************
 *
 * Transformation
 *
 *******************************/

// func (t *Tensor) Broadcasted(t2 *Tensor) []*Tensor {
// 	shape := t.node.dim.broadcastable(t2.node.dim)
// }

/*******************************
 *
 * Gradients
 *
 *******************************/

// func (t *Tensor) Backprop() {
// 	if t.grad == nil {
// 		t.grad = Vector([]float32{1})
// 	}

// 	flatten := func(t *Tensor) []*Tensor {
// 		visited := make(map[*Tensor]bool)
// 		var tensors []*Tensor
// 		var dfs func(*Tensor)
// 		dfs = func(_t *Tensor) {
// 			if _t.creator == nil {
// 				return
// 			}

// 			if visited[_t] {
// 				return
// 			}
// 			visited[_t] = true

// 			tensors = append(tensors, _t)
// 			for _, input := range _t.creator.inputs {
// 				dfs(input)
// 			}
// 		}

// 		dfs(t)
// 		return tensors
// 	}

// 	for _, tensor := range flatten(t) {
// 		grads := tensor.creator.backward(tensor.grad.node)
// 		for i := range grads {
// 			input := tensor.creator.inputs[i]
// 			grad := fromnode(grads[i])

// 			if input.grad == nil {
// 				input.grad = grad
// 			} else {
// 				y := input.grad.Add(grad)
// 				input.grad = y
// 			}
// 		}
// 	}
// }

func main() {
	t := Vector([]float32{1, 2})
	t2 := Vector([]float32{3, 4})
	t3 := t.Add(t2)

	// fmt.Println(t3)
	// t4 := t3.Div(Vector([]float32{10}))

	// t4.Backprop()

	// t.grad.Backprop()
	fmt.Println(t3.Materialize())
	// fmt.Println(t3.grad.Materialize())
	// fmt.Println(t2.grad.Materialize())
	// fmt.Println(t.grad.Materialize())
}
