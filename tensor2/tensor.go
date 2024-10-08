package main

import (
	"fmt"
	"os"
)

var debug bool

func initDebug() {
	debug = os.Getenv("DEBUG") == "1"
}

func init() {
	initDebug()
	initRunner()
}

type Tensor struct {
	graph    *graph    // knows how to construct this tensor.
	function *function // knows how to go backward the graph for differentiation.
	data     []float32 // actual data. Zero until materialized.
	grad     *Tensor   // gradient. Backprop() must be called to create.
}

func (t *Tensor) String() string {
	return fmt.Sprintf("%v", t.data)
}

/*******************************
 *
 * Tensor factory function
 *
 *******************************/

func New(data []float32) *Tensor {
	return &Tensor{graph: &graph{op: graphops.constant, constant: data}}
}

func empty() *Tensor {
	return &Tensor{}
}

func fromgraph(g *graph) *Tensor {
	return &Tensor{graph: g}
}

func fromfunc(d differentiable, inputs ...*Tensor) *Tensor {
	y := empty()
	y.function = &function{inputs: inputs, differentiable: d}

	graphs := make([]*graph, len(inputs))
	for i := range len(inputs) {
		graphs[i] = inputs[i].graph
	}
	y.graph = d.forward(graphs...)

	return y
}

/*******************************
 *
 * Tensor calculation
 *
 *******************************/

/*
 * Arithmetic
 */

func (t *Tensor) Recip() *Tensor {
	return fromfunc(&recip{}, t)
}

func (t *Tensor) Neg() *Tensor {
	return t.Mul(New([]float32{-1}))
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return fromfunc(&add{}, t, t2)
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	return t.Add(t2.Neg())
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return fromfunc(&mul{}, t, t2)
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	return t.Mul(t2.Recip())
}

/*******************************
 *
 * Gradients
 *
 *******************************/

func (t *Tensor) Backprop() {
	if t.grad == nil {
		t.grad = New([]float32{1})
	}

	flatten := func(t *Tensor) []*Tensor {
		visited := make(map[*Tensor]bool)
		var tensors []*Tensor
		var dfs func(*Tensor)
		dfs = func(_t *Tensor) {
			if _t.function == nil {
				return
			}

			if visited[_t] {
				return
			}
			visited[_t] = true

			tensors = append(tensors, _t)
			for _, input := range _t.function.inputs {
				dfs(input)
			}
		}

		dfs(t)
		return tensors
	}

	for _, tensor := range flatten(t) {
		grads := tensor.function.backward(tensor.grad.graph)
		for i := range grads {
			input := tensor.function.inputs[i]
			grad := fromgraph(grads[i])

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
	t := New([]float32{1})
	t2 := New([]float32{2})
	t3 := t.Add(t2)
	t4 := t3.Div(New([]float32{10}))

	// t4.Backprop()

	// t.grad.Backprop()
	fmt.Println(t4.Materialize())
	// fmt.Println(t3.grad.Materialize())
	// fmt.Println(t2.grad.Materialize())
	// fmt.Println(t.grad.Materialize())
}
