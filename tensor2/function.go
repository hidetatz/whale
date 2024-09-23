package main

type differentiable interface {
	forward(...*plan) *plan
	backward(*plan) []*plan
	String() string
}

type function struct {
	inputs         []*Tensor
	differentiable differentiable
}

func (f *function) backward(grad *plan) []*plan {
	return f.differentiable.backward(grad)
}

func applyfunc(d differentiable, inputs ...*Tensor) *Tensor {
	y := empty()
	y.function = &function{inputs: inputs, differentiable: d}

	plans := make([]*plan, len(inputs))
	for i := range len(inputs) {
		plans[i] = inputs[i].plan
	}
	y.plan = d.forward(plans...)

	return y
}

/*
 * differentiable
 */

type recip struct {
}

func (*recip) String() string { return " 1 / x" }

func (r *recip) forward(plans ...*plan) *plan {
	return &plan{op: ops.recip, src: []*plan{plans[0]}}
}

func (r *recip) backward(grad *plan) []*plan {
	return []*plan{
		// {op: ops.mul, src: []*plan{grad, m.y}},
		// {op: ops.mul, src: []*plan{grad, m.x}},
	}
}

type add struct{}

func (*add) String() string { return "+" }

func (*add) forward(plans ...*plan) *plan {
	return &plan{op: ops.add, src: []*plan{plans[0], plans[1]}}
}

func (*add) backward(grad *plan) []*plan { return []*plan{grad, grad} }

type mul struct {
	x, y *plan
}

func (*mul) String() string { return "*" }

func (m *mul) forward(plans ...*plan) *plan {
	m.x, m.y = plans[0], plans[1]
	return &plan{op: ops.mul, src: []*plan{plans[0], plans[1]}}
}

func (m *mul) backward(grad *plan) []*plan {
	return []*plan{
		{op: ops.mul, src: []*plan{grad, m.y}},
		{op: ops.mul, src: []*plan{grad, m.x}},
	}
}
