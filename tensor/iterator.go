package tensor

type Iterator struct {
	data     []float64
	offset   int
	isscalar bool
}

func (t *Tensor) Iterator() *Iterator {
	if t.IsScalar() {
		return &Iterator{data: []float64{t.AsScalar()}, offset: 0, isscalar: t.IsScalar()}
	}

	return &Iterator{data: t.Flatten(), offset: 0, isscalar: t.IsScalar()}
}

func (i *Iterator) HasNext() bool {
	if i.isscalar {
		return i.offset == 0
	}

	return i.offset < len(i.data)
}

func (i *Iterator) Next() (int, float64) {
	curoffset := i.offset
	if i.isscalar {
		i.offset++
		return curoffset, i.data[0]
	}

	v := i.data[i.offset]
	i.offset++
	return curoffset, v
}
