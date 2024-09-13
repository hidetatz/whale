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
	}

	panic("switch-case is not exhaustive!")
}

type _ops struct {
	constant op
	add      op
}

// Pseudo-namespacing
var ops = &_ops{
	1, 2,
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
	case ops.add:
		return fmt.Sprintf("{+ %v}", t.src)
	}

	panic("switch-case is not exhaustive!")
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

func main() {
	t := New([]float32{1, 2})
	t2 := New([]float32{3, 4})
	t3 := t.Add(t2)
	t4 := t3.Add(New([]float32{10, 10}))

	t4.Materialize()
	fmt.Println(t4.data)
}
