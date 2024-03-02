package tensor2

import "fmt"

type Iterator struct {
	indices [][]*IndexArg
	offset  int
	t       *Tensor
}

func (t *Tensor) Iterator() *Iterator {
	return &Iterator{indices: cartesianIdx(t.Shape), offset: 0, t: t}
}

func (i *Iterator) HasNext() bool {
	return i.offset < len(i.indices)
}

func (i *Iterator) Next() float64 {
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
