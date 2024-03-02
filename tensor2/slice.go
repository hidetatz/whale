package tensor2

import (
	"fmt"
	"slices"
)

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

func (t *Tensor) Index(args ...*IndexArg) (*Tensor, error) {
	if t.IsScalar() {
		return nil, fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(args) == 0 {
		return nil, fmt.Errorf("index accessor must not be empty")
	}

	if t.Ndim() < len(args) {
		return nil, fmt.Errorf("too many index accessors specified: %v", args)
	}

	// fill args to be the same length as ndim
	if len(args) < t.Ndim() {
		for _ = range t.Ndim() - len(args) {
			args = append(args, All())
		}
	}

	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })

	if advanced {
		return t.advancedIndex(args...)
	}

	return t.basicIndex(args...)
}

func (t *Tensor) basicIndex(args ...*IndexArg) (*Tensor, error) {
	newshape := make([]int, len(t.Shape))
	newstrides := make([]int, len(t.Strides))
	offset := t.offset
	for i, arg := range args {
		if 0 <= arg.i {
			if t.Shape[i]-1 < arg.i {
				return nil, fmt.Errorf("index out of bounds for axis %v with size %v", i, t.Shape[i])
			}

			offset += arg.i * t.Strides[i]
			newshape[i] = -1   // dummy value
			newstrides[i] = -1 // dummy value
			continue
		}

		if arg.s.step == 0 {
			return nil, fmt.Errorf("slice step must not be 0: %v", arg)
		}

		// Unlike Python, negative values are not allowed.
		if arg.s.step < 0 {
			arg.s.step = 1
		}

		if arg.s.start < 0 {
			arg.s.start = 0
		}

		if arg.s.end < 0 || t.Shape[i] < arg.s.end {
			arg.s.end = t.Shape[i]
		}

		if t.Shape[i] < arg.s.start || arg.s.end < arg.s.start {
			newshape[i] = 0
		} else {
			newshape[i] = (arg.s.end - arg.s.start + arg.s.step - 1) / arg.s.step
		}

		newstrides[i] = t.Strides[i] * arg.s.step

		if newshape[i] != 0 {
			offset += arg.s.start * t.Strides[i]
		}
	}

	deldummy := func(n int) bool { return n == -1 }
	newshape = slices.DeleteFunc(newshape, deldummy)
	newstrides = slices.DeleteFunc(newstrides, deldummy)

	return &Tensor{data: t.data, offset: offset, Shape: newshape, Strides: newstrides}, nil
}

func (t *Tensor) advancedIndex(args ...*IndexArg) (*Tensor, error) {
	return nil, nil
}

// func (t *Tensor) Slice(ss ...*Slice) (*Tensor, error) {
// 	if t.IsScalar() {
// 		return nil, fmt.Errorf("slice is not defined on scalar %v", t)
// 	}
//
// 	if len(ss) == 0 {
// 		return nil, fmt.Errorf("empty slices not allowed")
// 	}
//
// 	if len(t.Shape) < len(ss) {
// 		return nil, fmt.Errorf("too many slices specified: %v", ss)
// 	}
//
// 	// fill slices to be the same length as t.Shape
// 	if len(ss) < len(t.Shape) {
// 		n := len(t.Shape) - len(ss)
// 		for _ = range n {
// 			ss = append(ss, All())
// 		}
// 	}
//
// 	// In slicing, only the shape and strides are modified.
// 	// Shape is basically the same as t.Shape, but the actual data count calculated by
// 	// start/end/step is filled.
// 	// Strides is also the same as t.Strides, but it is multiplied by the step.
// 	newshape := make([]int, len(t.Shape))
// 	newstrides := make([]int, len(t.Strides))
// 	offset := t.offset
// 	for i, s := range ss {
// 		if s.Step == 0 {
// 			return nil, fmt.Errorf("slice step must not be 0: %v", ss)
// 		}
//
// 		// Unlike Python, negative values are not allowed.
// 		if s.Step < 0 {
// 			s.Step = 1
// 		}
//
// 		if s.Start < 0 {
// 			s.Start = 0
// 		}
//
// 		if s.End < 0 || t.Shape[i] < s.End {
// 			s.End = t.Shape[i]
// 		}
//
// 		if 0 <= s.At {
// 			if t.Shape[i]-1 < s.At {
// 				return nil, fmt.Errorf("index out of bounds for axis %v with size %v", i, t.Shape[i])
// 			}
//
// 			offset += s.At * t.Strides[i]
// 			newshape[i] = -1   // dummy value
// 			newstrides[i] = -1 // dummy value
// 			continue
// 		}
//
// 		if t.Shape[i] < s.Start || s.End < s.Start {
// 			newshape[i] = 0
// 		} else {
// 			newshape[i] = (s.End - s.Start + s.Step - 1) / s.Step
// 		}
//
// 		newstrides[i] = t.Strides[i] * s.Step
//
// 		if newshape[i] != 0 {
// 			offset += s.Start * t.Strides[i]
// 		}
// 	}
//
// 	deldummy := func(n int) bool { return n == -1 }
// 	newshape = slices.DeleteFunc(newshape, deldummy)
// 	newstrides = slices.DeleteFunc(newstrides, deldummy)
//
// 	return &Tensor{data: t.data, offset: offset, Shape: newshape, Strides: newstrides}, nil
// }
