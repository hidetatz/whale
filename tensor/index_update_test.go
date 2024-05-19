package tensor

import "testing"

func TestIndexPut(t *testing.T) {
	// Add
	tensor := Vector([]float32{1, 2, 3, 4, 5})
	tensor.IndexAdd([]*IndexArg{At(1)}, Scalar(2))
	mustEq(t, Vector([]float32{1, 4, 3, 4, 5}), tensor)

	tensor = Vector([]float32{1, 2, 3, 4, 5})
	tensor.IndexAdd([]*IndexArg{List(Vector([]float32{1, 1}))}, Scalar(2))
	mustEq(t, Vector([]float32{1, 4, 3, 4, 5}), tensor)

	// Sub
	tensor = New([][]float32{{1, 2}, {3, 4}})
	tensor.IndexSub([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float32{{-1, 0}, {3, 4}}), tensor)

	// Mul
	tensor = New([][]float32{{1, 2}, {3, 4}})
	tensor.IndexMul([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float32{{2, 4}, {3, 4}}), tensor)

	// Div
	tensor = New([][]float32{{1, 2}, {3, 4}})
	tensor.IndexDiv([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float32{{0.5, 1}, {3, 4}}), tensor)

	// Set
	tensor = New([][]float32{{1, 2}, {3, 4}})
	tensor.IndexSet([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float32{{2, 2}, {3, 4}}), tensor)
}
