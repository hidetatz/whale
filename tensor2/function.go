package main

type differentiable interface {
	forward(...*recipe) *recipe
	backward(*recipe) []*recipe
	String() string
}

type function struct {
	inputs         []*Tensor
	differentiable differentiable
}

func applyfunc(d differentiable, inputs ...*Tensor) *Tensor {
	y := empty()
	y.function = &function{inputs: inputs, differentiable: d}

	recipes := make([]*recipe, len(inputs))
	for i := range len(inputs) {
		recipes[i] = inputs[i].recipe
	}
	y.recipe = d.forward(recipes...)
	y.grad = empty()

	return y
}

/*
 * differentiable
 */

type add struct{}

func (*add) String() string { return "+" }

func (*add) forward(recipes ...*recipe) *recipe {
	return &recipe{op: ops.add, src: []*recipe{recipes[0], recipes[1]}}
}

func (*add) backward(grad *recipe) []*recipe { return []*recipe{grad, grad} }

type mul struct {
	x, y *recipe
}

func (m *mul) forward(recipes ...*recipe) *recipe {
	m.x, m.y = recipes[0], recipes[1]
	return &recipe{op: ops.mul, src: []*recipe{recipes[0], recipes[1]}}
}

func (m *mul) backward(grad *recipe) []*recipe {
	return []*recipe{
		{op: ops.mul, src: []*recipe{grad, m.y}},
		{op: ops.mul, src: []*recipe{grad, m.x}},
	}
}

func (*mul) String() string { return "*" }
