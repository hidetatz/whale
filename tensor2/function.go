package main

type op int

type _ops struct {
	constant, add, mul op
}

var ops = _ops{
	constant: 0, add: 1, mul: 2,
}

type calculation struct {
	do           func(inputs ...*Tensor) *Tensor
	differential func(inputs []*Tensor, grad *Tensor) []*Tensor
}

type _calculations struct {
	add *calculation
	mul *calculation
}

var calculations = &_calculations{
	add: &calculation{
		do: func(inputs ...*Tensor) *Tensor {
			return &Tensor{op: ops.add, inputs: []*Tensor{inputs[0], inputs[1]}}
		},
		differential: func(_ []*Tensor, grad *Tensor) []*Tensor {
			return []*Tensor{grad, grad}
		},
	},
	mul: &calculation{
		do: func(inputs ...*Tensor) *Tensor {
			return &Tensor{op: ops.mul, inputs: []*Tensor{inputs[0], inputs[1]}}
		},
		differential: func(inputs []*Tensor, grad *Tensor) []*Tensor {
			return []*Tensor{
				{op: ops.mul, inputs: []*Tensor{grad, inputs[1]}},
				{op: ops.mul, inputs: []*Tensor{grad, inputs[0]}},
			}
		},
	},
}
