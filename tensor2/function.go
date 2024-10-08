package main

type differentiable interface {
	forward(...*graph) *graph
	backward(*graph) []*graph
	String() string
}

type function struct {
	inputs         []*Tensor
	differentiable differentiable
}

func (f *function) backward(grad *graph) []*graph {
	return f.differentiable.backward(grad)
}

/*
 * differentiables
 */

type recip struct {
}

func (*recip) String() string { return " 1 / x" }

func (r *recip) forward(graphs ...*graph) *graph {
	return &graph{op: graphops.recip, input: []*graph{graphs[0]}}
}

func (r *recip) backward(grad *graph) []*graph {
	return []*graph{
		// {op: ops.mul, src: []*graph{grad, m.y}},
		// {op: ops.mul, src: []*graph{grad, m.x}},
	}
}

type add struct{}

func (*add) String() string { return "+" }

func (*add) forward(graphs ...*graph) *graph {
	return &graph{op: graphops.add, input: []*graph{graphs[0], graphs[1]}}
}

func (*add) backward(grad *graph) []*graph { return []*graph{grad, grad} }

type mul struct {
	x, y *graph
}

func (*mul) String() string { return "*" }

func (m *mul) forward(graphs ...*graph) *graph {
	m.x, m.y = graphs[0], graphs[1]
	return &graph{op: graphops.mul, input: []*graph{graphs[0], graphs[1]}}
}

func (m *mul) backward(grad *graph) []*graph {
	return []*graph{
		{op: graphops.mul, input: []*graph{grad, m.y}},
		{op: graphops.mul, input: []*graph{grad, m.x}},
	}
}
