package whale

import (
	"cmp"
	"slices"
)

type function struct {
	inputs     []*Variable
	outputs    []*Variable
	generation int
	operation  Operation
}

func NewFunction(op Operation) *function {
	return &function{operation: op}
}

func (f *function) forward(inputs ...*Variable) []*Variable {
	outputs := f.operation.Forward(inputs...)
	if EnableBackprop {
		f.inputs = inputs
		for _, o := range outputs {
			o.SetCreator(f)
		}
		f.generation = getMaxGen(inputs)
		f.outputs = outputs
	}
	return outputs
}

func (f *function) String() string {
	return f.operation.String()
}

func getMaxGen(vs []*Variable) int {
	return slices.MaxFunc(vs, func(a, b *Variable) int {
		return cmp.Compare(a.generation, b.generation)
	}).generation
}
