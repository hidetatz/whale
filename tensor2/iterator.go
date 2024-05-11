package tensor2

type Iterator struct {
	data   []float64
	offset int
	t      *Tensor
}

func (t *Tensor) Iterator() *Iterator {
	if t.IsScalar() {
		return &Iterator{data: nil, offset: 0, t: t}
	}

	return &Iterator{data: t.Flatten(), offset: 0, t: t}
}

func (i *Iterator) HasNext() bool {
	if i.t.IsScalar() {
		return i.offset == 0
	}

	return i.offset < len(i.data)
}

func (i *Iterator) Next() (int, float64) {
	curoffset := i.offset
	if i.t.IsScalar() {
		i.offset++
		return curoffset, i.t.AsScalar()
	}

	v := i.data[i.offset]
	i.offset++
	return curoffset, v
}
