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

func (t *Tensor) Squeeze(axes ...int) (*Tensor, error) {
	for _, axis := range axes {
		if t.Ndim() < axis {
			return nil, fmt.Errorf("axis out of bounds: %v for %v tensor", axis, t.Ndim())
		}

		if t.Shape[axis] != 1 {
			return nil, fmt.Errorf("non-1 axis is specified: %v for axis whose size is %v", axis, t.Shape[axis])
		}
	}

	newshape := []int{}
	newstrides := []int{}
	for i := range t.Shape {
		if t.Shape[i] != 1 {
			newshape = append(newshape, t.Shape[i])
			newstrides = append(newstrides, t.Strides[i])
			continue
		}

		if len(axes) != 0 && !slices.Contains(axes, i) {
			newshape = append(newshape, t.Shape[i])
			newstrides = append(newstrides, t.Strides[i])
		}
	}

	return &Tensor{data: t.data, offset: t.offset, Shape: newshape, Strides: newstrides}, nil
}

func (t *Tensor) BroadcastTo(shape ...int) (*Tensor, error) {
	// validation
	if len(t.Shape) > len(shape) {
		return nil, fmt.Errorf("invalid desired shape")
	}

	// initialized with 0
	delta := len(shape) - len(t.Shape)
	newstrides := make([]int, len(shape))
	for i := t.Ndim() - 1; 0 <= i; i-- {
		if shape[delta+i] == t.Shape[i] && t.Shape[i] == 1 {
			newstrides[delta+i] = 0
			continue
		}

		if shape[delta+i] == t.Shape[i] && t.Shape[i] != 1 {
			newstrides[delta+i] = t.Strides[i]
			continue
		}

		if t.Shape[i] != 1 {
			return nil, fmt.Errorf("cannot broadcast: original shape is %v, target is %v", t.Shape, shape)
		}

		newstrides[delta+i] = 0
	}

	return &Tensor{data: t.data, offset: t.offset, Shape: shape, Strides: newstrides}, nil
}
