package whale

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

func getMaxGen(vs []*Variable) int {
	max := 0
	for _, v := range vs {
		if v.generation > max {
			max = v.generation
		}
	}
	return max
}
