package tensor2

import "fmt"

type slice struct {
	start, end, step int
}

func (s *slice) tidy(dimlen int) error {
	if s.step == 0 {
		return fmt.Errorf("slice step must not be 0: %v", s)
	}

	// Unlike Python, negative values are not allowed.
	if s.step < 0 {
		s.step = 1
	}

	if s.start < 0 {
		s.start = 0
	}

	if s.end < 0 || dimlen < s.end {
		s.end = dimlen
	}

	return nil
}

func (s *slice) size() int {
	if s.end <= s.start {
		return 0
	}
	return (s.end - s.start + s.step - 1) / s.step
}

func (s *slice) indices() []int {
	r := make([]int, s.size())
	for i := range r {
		r[i] = s.start + i*s.step
	}
	return r
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

func (s *slice) Copy() *slice {
	return &slice{start: s.start, end: s.end, step: s.step}
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

// FromBy creates a tensor slicing accessor like "x[start::step]".
func FromBy(start, step int) *IndexArg {
	return &IndexArg{s: &slice{start: start, end: -1, step: 1}, typ: _slice}
}

// ToBy creates a tensor slicing accessor like "x[:end:step]".
func ToBy(end, step int) *IndexArg {
	return &IndexArg{s: &slice{start: -1, end: end, step: 1}, typ: _slice}
}

// FromToBy creates a tensor slicing accessor like "x[start:end:step]".
func FromToBy(start, end, step int) *IndexArg {
	return &IndexArg{s: &slice{start: start, end: end, step: step}, typ: _slice}
}

// All creates a tensor slicing accessor like "x[:]".
func All() *IndexArg { return &IndexArg{s: &slice{start: -1, end: -1, step: -1}, typ: _slice} }

// List creates a tensor accessor like "x[[[1, 0], [2, 3]]]" on a tensor "New([][]float64{{1, 0}, {2, 3}})"
func List(t *Tensor) *IndexArg { return &IndexArg{t: t, typ: _tensor} }

func (a *IndexArg) String() string {
	switch a.typ {
	case _int:
		return fmt.Sprintf("%d", a.i)
	case _slice:
		return fmt.Sprintf("%v", a.s)
	case _tensor:
		return a.t.asPythonListString()
	}
	panic(fmt.Sprintf("unknown typ in IndexArg: %v", a.typ))
}

func (a *IndexArg) numpyIndexString() string {
	if a.typ == _tensor {
		return a.t.asPythonListString()
	}

	return a.String()
}

func (a *IndexArg) Copy() *IndexArg {
	return &IndexArg{
		i:   a.i,
		s:   a.s.Copy(),
		t:   a.t.Copy(),
		typ: a.typ,
	}
}
