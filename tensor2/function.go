package main

type differentiable interface {
	forward(...*node) *node
	backward(*node) []*node
	String() string
}

type function struct {
	inputs         []*Tensor
	differentiable differentiable
}

func (f *function) backward(grad *node) []*node {
	return f.differentiable.backward(grad)
}

/*
 * differentiables
 */

type recip struct {
}

func (*recip) String() string { return " 1 / x" }

func (r *recip) forward(graphs ...*node) *node {
	return &node{op: nodeops.recip, input: []*node{graphs[0]}}
}

func (r *recip) backward(grad *node) []*node {
	return []*node{
		// {op: ops.mul, src: []*graph{grad, m.y}},
		// {op: ops.mul, src: []*graph{grad, m.x}},
	}
}

type add struct{}

func (*add) String() string { return "+" }

func (*add) forward(graphs ...*node) *node {
	return &node{op: nodeops.add, input: []*node{graphs[0], graphs[1]}}
}

func (*add) backward(grad *node) []*node { return []*node{grad, grad} }

type mul struct {
	x, y *node
}

func (*mul) String() string { return "*" }

func (m *mul) forward(graphs ...*node) *node {
	m.x, m.y = graphs[0], graphs[1]
	return &node{op: nodeops.mul, input: []*node{graphs[0], graphs[1]}}
}

func (m *mul) backward(grad *node) []*node {
	return []*node{
		{op: nodeops.mul, input: []*node{grad, m.y}},
		{op: nodeops.mul, input: []*node{grad, m.x}},
	}
}
