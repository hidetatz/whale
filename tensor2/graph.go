package main

import (
	"fmt"
)

type graphop int

func (op graphop) String() string {
	switch op {
	case graphops.constant:
		return "const"

	case graphops.recip:
		return "1/x"

	case graphops.add:
		return "+"

	case graphops.mul:
		return "*"
	}

	panic("switch-case is not exhaustive!")
}

type _graphops struct {
	constant graphop
	recip    graphop
	add      graphop
	mul      graphop
}

// Pseudo-namespacing
var graphops = &_graphops{
	1, 2, 3, 4,
}

type graph struct {
	op       graphop
	constant []float32
	input    []*graph
}

func (g *graph) String() string {
	switch g.op {
	case graphops.constant:
		return fmt.Sprintf("%v", g.constant)
	default:
		return fmt.Sprintf("%v", g.op)
	}
}
