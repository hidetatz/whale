package main

type operation int

const (
	op_const operation = iota + 1

	op_recip
	op_add
	op_mul

	op_expand
)

// forward/backward
type calculation interface {
	do(inputs ...*Tensor) *Tensor
	differential(grad *Tensor) []*Tensor
}

func (op operation) String() string {
	switch op {
	case op_const:
		return "const"

	case op_recip:
		return "1 / x"

	case op_add:
		return "+"

	case op_mul:
		return "*"

	case op_expand:
		return "expand"

	default:
		panic("switch-case is not exhaustive!")
	}
}

/*
 * Arithmetic
 */

type calcRecip struct {
	ret *Tensor
}

func (c *calcRecip) do(inputs ...*Tensor) *Tensor {
	c.ret = &Tensor{op: op_recip, inputs: []*Tensor{inputs[0]}, dim: newdim(inputs[0].dim.shape...)}
	return c.ret
}

func (c *calcRecip) differential(grad *Tensor) []*Tensor {
	// reciprocal' is (-1/(x^2)), because ret is (1/x), so this returns (-1*ret*ret)
	return []*Tensor{Scalar(-1).Mul(c.ret).Mul(c.ret)}
}

type calcAdd struct{}

func (c *calcAdd) do(inputs ...*Tensor) *Tensor {
	return &Tensor{op: op_add, inputs: []*Tensor{inputs[0], inputs[1]}, dim: newdim(inputs[0].dim.shape...)}
}

func (c *calcAdd) differential(grad *Tensor) []*Tensor {
	return []*Tensor{grad, grad}
}

type calcMul struct {
	x, y *Tensor
}

func (c *calcMul) do(inputs ...*Tensor) *Tensor {
	return &Tensor{op: op_mul, inputs: []*Tensor{inputs[0], inputs[1]}, dim: newdim(inputs[0].dim.shape...)}
}

func (c *calcMul) differential(grad *Tensor) []*Tensor {
	return []*Tensor{
		{op: op_mul, inputs: []*Tensor{grad, c.y}},
		{op: op_mul, inputs: []*Tensor{grad, c.x}},
	}
}

/*
 * Transformation
 */

type calcExpand struct {
	shape, origshape []int
}

func (c *calcExpand) do(inputs ...*Tensor) *Tensor {
	c.origshape = inputs[0].dim.shape
	return &Tensor{op: op_expand, inputs: []*Tensor{inputs[0]}, dim: inputs[0].dim.expand(c.shape...)}
}

func (c *calcExpand) differential(grad *Tensor) []*Tensor {
	// todo: sum using orig shape
	panic("not implemented")
}
