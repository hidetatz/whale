package main

type nodeop int

func (op nodeop) String() string {
	switch op {
	case nodeops.constant:
		return "const"

	case nodeops.recip:
		return "1/x"

	case nodeops.add:
		return "+"

	case nodeops.mul:
		return "*"
	}

	panic("switch-case is not exhaustive!")
}

type _nodeops struct {
	constant nodeop
	recip    nodeop
	add      nodeop
	mul      nodeop
}

var nodeops = &_nodeops{
	1, 2, 3, 4,
}

// computation graph node.
type node struct {
	op       nodeop
	constant []float32
	input    []*node
	dim      *dimension
}

func (n *node) String() string {
	return n.op.String()
}
