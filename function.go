package whale

import (
	"cmp"
	"slices"
)

type function struct {
	inputs     []*Variable
	outputs    []*Variable
	generation int
	op         Op
}

func NewFunction(op Op) *function {
	return &function{op: op}
}

func (f *function) forward(inputs ...*Variable) ([]*Variable, error) {
	outputs, err := f.op.Forward(inputs...)
	if err != nil {
		return nil, err
	}

	if EnableBackprop {
		f.inputs = inputs
		for _, o := range outputs {
			o.SetCreator(f)
		}
		f.generation = getMaxGen(inputs)
		f.outputs = outputs
	}
	return outputs, nil
}

func (f *function) String() string {
	return f.op.String()
}

func getMaxGen(vs []*Variable) int {
	return slices.MaxFunc(vs, func(a, b *Variable) int {
		return cmp.Compare(a.generation, b.generation)
	}).generation
}
