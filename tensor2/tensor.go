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

type op int

func (op op) String() string {
	switch op {
	case ops.constant:
		return "const"

	case ops.add:
		return "+"

	case ops.mul:
		return "*"
	}

	panic("switch-case is not exhaustive!")
}

type _ops struct {
	constant op
	add      op
	mul      op
}

// Pseudo-namespacing
var ops = &_ops{
	1, 2, 3,
}

type plan struct {
	op       op
	constant []float32
	src      []*plan
}

func (p *plan) String() string {
	switch p.op {
	case ops.constant:
		return fmt.Sprintf("%v", p.constant)
	default:
		return fmt.Sprintf("%v", p.op)
	}
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

func empty() *Tensor {
	return &Tensor{}
}

func New(data []float32) *Tensor {
	return &Tensor{plan: &plan{op: ops.constant, constant: data}}
}

func fromplan(p *plan) *Tensor {
	return &Tensor{plan: p}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return applyfunc(&add{}, t, t2)
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return applyfunc(&mul{}, t, t2)
}

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
	t4 := t3.Mul(New([]float32{10}))

	t4.Backprop()

	fmt.Println(t4.grad.Materialize())
	fmt.Println(t3.grad.Materialize())
	fmt.Println(t2.grad.Materialize())
	fmt.Println(t.grad.Materialize())
}
