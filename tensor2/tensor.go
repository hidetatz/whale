package main

import (
	"fmt"
)

func init() {
	initRunner()
}

type op int

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

func empty(op op) *Tensor {
	return &Tensor{op: op}
}

func New(data []float32) *Tensor {
	return &Tensor{op: ops.constant, data: data}
}

type task struct {
	op     op
	data   []float32
	inputs []int
}

func (t *Tensor) toposort() []*task {
	// todo: do toposort
	task0 := &task{op: t.src[0].src[0].op, data: t.src[0].src[0].data}
	task1 := &task{op: t.src[0].src[1].op, data: t.src[0].src[1].data}
	task2 := &task{op: t.src[0].op, inputs: []int{0, 1}}
	task3 := &task{op: t.src[1].op, data: t.src[1].data}
	task4 := &task{op: t.op, inputs: []int{2, 3}}
	return []*task{task0, task1, task3, task2, task4}
}

func (t *Tensor) Materialize() []float32 {
	tasks := t.toposort()
	result := runner.run(tasks)
	t.data = result
	return t.data
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
