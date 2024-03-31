package tensor2

import "fmt"

type Iterator struct {
	indices [][]*IndexArg
	offset  int
	t       *Tensor
}

func (t *Tensor) Iterator() *Iterator {
	if t.IsScalar() {
		return &Iterator{indices: nil, offset: 0, t: t}
	}

	return &Iterator{indices: cartesianIdx(t.Shape), offset: 0, t: t}
}

func (i *Iterator) HasNext() bool {
	if i.t.IsScalar() {
		return i.offset == 0
	}

	return i.offset < len(i.indices)
}

func (i *Iterator) Next() float64 {
	if i.t.IsScalar() {
		i.offset++
		return i.t.AsScalar()
	}

	idx := i.indices[i.offset]
	s, err := i.t.Index(idx...)
	if err != nil {
		panic(fmt.Errorf("Next: index access problem: %v", idx))
	}

	if !s.IsScalar() {
		panic("Next: non scalar")
	}

	i.offset++
	return s.AsScalar()
}
