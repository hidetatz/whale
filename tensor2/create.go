package main

import (
	"fmt"
)

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
	recip    op
	add      op
	mul      op
}

// Pseudo-namespacing
var ops = &_ops{
	1, 2, 3, 4,
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
