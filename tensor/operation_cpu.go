package tensor

import "math"

func Pow(t *Tensor, y float64) *Tensor {
	t2 := t.Copy()
	for i := range t2.data {
		t2.data[i] = math.Pow(t2.data[i], y)
	}
	return t2
}
