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

		if err := arg.s.tidy(t.Shape[i]); err != nil {
			return nil, err
		}

		if t.Shape[i] < arg.s.start || arg.s.end < arg.s.start {
			newshape[i] = 0
		} else {
			newshape[i] = arg.s.size()
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

func (t *Tensor) basicIndexUpdate(fn func(float64) float64, args ...*IndexArg) error {
	t2, err := t.basicIndex(args...)
	if err != nil {
		return err
	}

	for _, idx := range t2.rawIndices() {
		t.data[idx] = fn(t.data[idx])
	}

	return nil
}
