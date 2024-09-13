package main

import (
	"fmt"
)

func init() {
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

type Tensor struct {
	op   op
	src  []*Tensor
	data []float32
}

func (t *Tensor) String() string {
	switch t.op {
	case ops.constant:
		return fmt.Sprintf("%v", t.data)
	default:
		return fmt.Sprintf("{%v %v}", t.op, t.src)
	}
}

func empty(op op) *Tensor {
	return &Tensor{op: op}
}

func New(data []float32) *Tensor {
	return &Tensor{op: ops.constant, data: data}
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	y := empty(ops.add)
	y.src = []*Tensor{t, t2}
	return y
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	y := empty(ops.mul)
	y.src = []*Tensor{t, t2}
	return y
}

func main() {
	t := New([]float32{1, 2})
	t2 := New([]float32{3, 4})
	t3 := t.Add(t2)
	t4 := t3.Mul(New([]float32{10, 10}))

	// t5 := New([]float32{1, 2})
	// t6 := New([]float32{3, 4})
	// t7 := t5.Add(t6)
	// t8 := t7.Add(New([]float32{10, 10}))

	// t9 := t8.Add(t4)
	// t9.Materialize()
	t4.Materialize()
	fmt.Println(t4.data)
}
