package tensor2

import (
	"fmt"
	"slices"
)

func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if product(shape) != t.Size() {
		return nil, fmt.Errorf("cannot reshape size %v tensor into %v", t.Size(), shape)
	}

	// reshape shares original tensor data/offset, only shape and strides are modified.
	t2 := &Tensor{data: t.data, offset: t.offset}
	t2.Shape = shape
	strides := make([]int, len(shape))
	for i := range shape {
		strides[i] = product(shape[i+1:])
	}
	t2.Strides = strides

	return t2, nil
}

func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
	if t.IsScalar() {
		return t, nil
	}

	seq := func(from, to int) []int {
		r := make([]int, to-from)
		for i := range r {
			r[i] = from + i
		}
		return r
	}

	// check

	if len(axes) == 0 {
		// if axes is empty, create [0, 1, 2...] slice and reverses it
		axes = seq(0, t.Ndim())
		slices.Reverse(axes)
	} else {
		// else, axes must be an arbitrarily sorted slice of [0, 1, 2, ... dim].
		copied := make([]int, len(axes))
		copy(copied, axes)
		slices.Sort(copied)
		if !slices.Equal(copied, seq(0, t.Ndim())) {
			return nil, fmt.Errorf("axes don't much: %v for shape %v", axes, t.Shape)
		}
	}

	// do transpose

	t2 := &Tensor{data: t.data, offset: t.offset, Shape: make([]int, t.Ndim()), Strides: make([]int, t.Ndim())}
	for i, axis := range axes {
		t2.Shape[i] = t.Shape[axis]
		t2.Strides[i] = t.Strides[axis]
	}

	return t2, nil
}
