package tensor

import "math"

func PowCPU(t *Tensor, y float64) *Tensor {
	t2 := t.Copy()
	for i := range t2.data {
		t2.data[i] = math.Pow(t2.data[i], y)
	}
	return t2
}

func ExpCPU(t *Tensor) *Tensor {
	t2 := t.Copy()
	for i := range t2.data {
		t2.data[i] = math.Exp(t2.data[i])
	}
	return t2
}
