package tensor

func (t *Tensor) Bool(f func(f float32) bool) *Tensor {
	d := make([]float32, t.Size())

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
