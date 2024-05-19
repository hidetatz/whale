package tensor

func (t *Tensor) Bool(f func(f float64) bool) *Tensor {
	d := make([]float64, t.Size())

	iter := t.Iterator()
	for iter.HasNext() {
		i, v := iter.Next()
		if f(v) {
			d[i] = 1
		} else {
			d[i] = 0
		}
	}

	return NdShape(d, copySlice(t.Shape)...)
}
