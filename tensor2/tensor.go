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
	plan     *plan     // knows how to construct this tensor.
	function *function // knows how to create/differentiate plan.
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
	return &Tensor{plan: &plan{op: ops.constant, constant: data}}
}

func empty() *Tensor {
	return &Tensor{}
}

func fromplan(p *plan) *Tensor {
	return &Tensor{plan: p}
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
	return applyfunc(&recip{}, t)
}

func (t *Tensor) Neg() *Tensor {
	return applyfunc(&mul{}, t, New([]float32{-1}))
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return applyfunc(&add{}, t, t2)
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	return applyfunc(&add{}, t, t2.Neg())
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return applyfunc(&mul{}, t, t2)
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	return applyfunc(&mul{}, t, t2.Recip())
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
		grads := tensor.function.backward(tensor.grad.plan)
		for i := range grads {
			input := tensor.function.inputs[i]
			grad := fromplan(grads[i])

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
