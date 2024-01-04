package tensor

type Iterator struct {
	t    *Tensor
	axis int
	idx  int
}

func (i *Iterator) HasNext() bool {
	if i.axis == -1 {
		return i.idx == 0
	}

	strides := i.t.strides()
	retcnt := strides[i.axis]
	offset := i.idx * retcnt
	return offset < len(i.t.Data)
}

func (i *Iterator) Next() []float64 {
	if i.axis == -1 {
		i.idx++
		return i.t.Data
	}

	strides := i.t.strides()
	retcnt := strides[i.axis]
	offset := i.idx * retcnt
	i.idx++
	return i.t.Data[offset : offset+retcnt]
}
