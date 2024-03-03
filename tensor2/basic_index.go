package tensor2

import (
	"fmt"
	"slices"
)

func (t *Tensor) basicIndex(args ...*IndexArg) (*Tensor, error) {
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

	newshape := make([]int, len(t.Shape))
	newstrides := make([]int, len(t.Strides))
	offset := t.offset
	for i, arg := range args {
		if arg.typ == _int {
			if t.Shape[i]-1 < arg.i {
				return nil, fmt.Errorf("index out of bounds for axis %v with size %v", i, t.Shape[i])
			}

			offset += arg.i * t.Strides[i]
			newshape[i] = -1   // dummy value
			newstrides[i] = -1 // dummy value
			continue
		}

		// coming here means the arg type is slice

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
