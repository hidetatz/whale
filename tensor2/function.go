package main

type operation int

const (
	// underscore makes it easier to read for me
	op_const operation = iota + 1

	op_add
	op_mul

	// transform
	op_expand
)

// calculation context
type context struct {
	inputs []*Tensor
	params map[string]any
}

func newctx() *context {
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

// forward/backward
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
			return &Tensor{op: op_add, inputs: []*Tensor{inputs[0], inputs[1]}, dim: newdim(inputs[0].dim.shape...)}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			return []*Tensor{grad, grad}
		},
	},

	mul: &calc{
		do: func(ctx *context, inputs ...*Tensor) *Tensor {
			return &Tensor{op: op_mul, inputs: []*Tensor{inputs[0], inputs[1]}, dim: newdim(inputs[0].dim.shape...)}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			return []*Tensor{
				{op: op_mul, inputs: []*Tensor{grad, ctx.inputs[1]}},
				{op: op_mul, inputs: []*Tensor{grad, ctx.inputs[0]}},
			}
		},
	},

	/*
	 * Transformation
	 */

	expand: &calc{
		do: func(ctx *context, inputs ...*Tensor) *Tensor {
			ctx.set("original_shape", inputs[0].dim.shape)
			return &Tensor{op: op_expand, inputs: []*Tensor{inputs[0]}, dim: inputs[0].dim.expand(ctx.getshape()...)}
		},
		differential: func(ctx *context, grad *Tensor) []*Tensor {
			// todo: sum using orig shape
			panic("not implemented")
		},
	},
}
