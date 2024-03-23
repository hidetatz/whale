package tensor2

import "testing"

func TestIndexWrite(t *testing.T) {
	// Add
	tensor := Vector([]float64{1, 2, 3, 4, 5})
	tensor.IndexAdd(2, At(1))
	mustEq(t, Vector([]float64{1, 4, 3, 4, 5}), tensor)

	// Sub
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexSub(2, At(0))
	mustEq(t, Must(New([][]float64{{-1, 0}, {3, 4}})), tensor)

	// Mul
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexMul(2, At(0))
	mustEq(t, Must(New([][]float64{{2, 4}, {3, 4}})), tensor)

	// Div
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexDiv(2, At(0))
	mustEq(t, Must(New([][]float64{{0.5, 1}, {3, 4}})), tensor)

	// Set
	tensor = Must(New([][]float64{{1, 2}, {3, 4}}))
	tensor.IndexSet(2, At(0))
	mustEq(t, Must(New([][]float64{{2, 2}, {3, 4}})), tensor)
}
