package main

type op int

type _ops struct {
	constant, add, mul, expand op
}

var ops = _ops{
	constant: 0, add: 1, mul: 2, expand: 3,
}

// calculation context
type context struct {
	inputs []*Tensor
	params map[string]any
}

func newcontext() *context {
	return &context{params: map[string]any{}}
}

func (c *context) set(key string, val any) *context {
	c.params[key] = val
	return c
}

func (c *context) get(key string) any {
	return c.params[key]
}

func (c *context) setshape(shape ...int) *context {
	return c.set("shape", shape)
}

func (c *context) getshape() []int {
	return c.get("shape").([]int)
}

type calc struct {
	do           func(ctx *context, inputs ...*Tensor) *Tensor
	differential func(ctx *context, grad *Tensor) []*Tensor
}

/*
 * Arithmetic
 */

type _calculations struct {
	add    *calc
	mul    *calc
	expand *calc
}

var calculations = &_calculations{

	/*
	 * Arithmetic
	 */

	add: &calc{
		do: func(ctx *context, inputs ...*Tensor) *Tensor {
			return &Tensor{op: ops.add, inputs: []*Tensor{inputs[0], inputs[1]}}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			return []*Tensor{grad, grad}
		},
	},

	mul: &calc{
		do: func(ctx *context, inputs ...*Tensor) *Tensor {
			return &Tensor{op: ops.mul, inputs: []*Tensor{inputs[0], inputs[1]}}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			return []*Tensor{
				{op: ops.mul, inputs: []*Tensor{grad, ctx.inputs[1]}},
				{op: ops.mul, inputs: []*Tensor{grad, ctx.inputs[0]}},
			}
		},
	},

	/*
	 * Transformation
	 */

	expand: &calc{
		do: func(ctx *context, inputs ...*Tensor) *Tensor {
			ctx.set("original_shape", inputs[0].dim.shape)
			return &Tensor{op: ops.expand, inputs: []*Tensor{inputs[0]}, dim: newdimension(ctx.getshape()...)}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			// todo: sum using orig shape
			panic("not implemented")
		},
	},
}
