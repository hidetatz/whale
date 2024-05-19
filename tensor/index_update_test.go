package tensor

import "testing"

func TestIndexPut(t *testing.T) {
	// Add
	tensor := Vector([]float64{1, 2, 3, 4, 5})
	tensor.IndexAdd([]*IndexArg{At(1)}, Scalar(2))
	mustEq(t, Vector([]float64{1, 4, 3, 4, 5}), tensor)

	tensor = Vector([]float64{1, 2, 3, 4, 5})
	tensor.IndexAdd([]*IndexArg{List(Vector([]float64{1, 1}))}, Scalar(2))
	mustEq(t, Vector([]float64{1, 4, 3, 4, 5}), tensor)

	// Sub
	tensor = New([][]float64{{1, 2}, {3, 4}})
	tensor.IndexSub([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float64{{-1, 0}, {3, 4}}), tensor)

	// Mul
	tensor = New([][]float64{{1, 2}, {3, 4}})
	tensor.IndexMul([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float64{{2, 4}, {3, 4}}), tensor)

	// Div
	tensor = New([][]float64{{1, 2}, {3, 4}})
	tensor.IndexDiv([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float64{{0.5, 1}, {3, 4}}), tensor)

	// Set
	tensor = New([][]float64{{1, 2}, {3, 4}})
	tensor.IndexSet([]*IndexArg{At(0)}, Scalar(2))
	mustEq(t, New([][]float64{{2, 2}, {3, 4}}), tensor)
}
