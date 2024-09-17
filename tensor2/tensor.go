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

type recipe struct {
	op       op
	constant []float32
	src      []*recipe
}

func (r *recipe) String() string {
	switch r.op {
	case ops.constant:
		return fmt.Sprintf("%v", r.constant)
	default:
		return fmt.Sprintf("%v", r.op)
	}
}

type Tensor struct {
	function *function
	recipe   *recipe
	data     []float32
	grad     *Tensor
}

func (t *Tensor) String() string {
	return fmt.Sprintf("%v", t.data)
}

func empty() *Tensor {
	return &Tensor{}
}

func New(data []float32) *Tensor {
	return &Tensor{recipe: &recipe{op: ops.constant, constant: data}, grad: empty()}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return applyfunc(&add{}, t, t2)
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return applyfunc(&mul{}, t, t2)
}

func (t *Tensor) Backprop() {
	t.grad = New([]float32{1})

	recipes := t.function.differentiable.backward(t.grad.recipe)
	for i := range recipes {
		t.function.inputs[i].grad.recipe = recipes[i]
	}

	recipes = t.function.inputs[0].function.differentiable.backward(t.function.inputs[0].grad.recipe)
	for i := range recipes {
		t.function.inputs[0].function.inputs[i].grad.recipe = recipes[i]
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
