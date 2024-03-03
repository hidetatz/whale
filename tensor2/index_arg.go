package tensor2

import "fmt"

type slice struct {
	start, end, step int
}

func (s *slice) String() string {
	r := ""

	if 0 <= s.start {
		r += fmt.Sprintf("%d", s.start)
	}
	r += ":"
	if 0 <= s.end {
		r += fmt.Sprintf("%d", s.end)
	}
	r += ":"
	if 0 <= s.step {
		r += fmt.Sprintf("%d", s.step)
	}
	return r
}

type IndexArg struct {
	i   int
	s   *slice
	t   *Tensor
	typ int
}

const (
	_int = iota + 1
	_slice
	_tensor
)

// At creates a tensor index like "x[i]".
func At(i int) *IndexArg { return &IndexArg{i: i, typ: _int} }

// From creates a tensor accessor like "x[start::]".
func From(start int) *IndexArg {
	return &IndexArg{s: &slice{start: start, end: -1, step: 1}, typ: _slice}
}

// To creates a tensor slicing accessor like "x[:end:]".
func To(end int) *IndexArg { return &IndexArg{s: &slice{start: -1, end: end, step: 1}, typ: _slice} }

// By creates a tensor slicing accessor like "x[::step]".
func By(step int) *IndexArg { return &IndexArg{s: &slice{start: -1, end: -1, step: step}, typ: _slice} }

// FromTo creates a tensor slicing accessor like "x[start:end]".
func FromTo(start, end int) *IndexArg {
	return &IndexArg{s: &slice{start: start, end: end, step: 1}, typ: _slice}
}

// FromToBy creates a tensor slicing accessor like "x[start:end:step]".
func FromToBy(start, end, step int) *IndexArg {
	return &IndexArg{s: &slice{start: start, end: end, step: step}, typ: _slice}
}

// All creates a tensor slicing accessor like "x[:]".
func All() *IndexArg { return &IndexArg{s: &slice{start: -1, end: -1, step: -1}, typ: _slice} }

// List creates a tensor accessor on tensor "New([][]float64{{1, 0}, {2, 3}})" like "x[[[1, 0], [2, 3]]]"
func List(t *Tensor) *IndexArg { return &IndexArg{t: t, typ: _tensor} }

func (i *IndexArg) String() string {
	switch i.typ {
	case _int:
		return fmt.Sprintf("%d", i.i)
	case _slice:
		return fmt.Sprintf("%v", i.s)
	case _tensor:
		return fmt.Sprintf("%v", i.t)
	}
	panic(fmt.Sprintf("unknown typ in IndexArg: %v", i.typ))
}
