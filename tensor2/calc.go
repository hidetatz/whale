package tensor2

import (
	"fmt"
	"slices"
)

func (t *Tensor) Sum(keepdims bool, axes ...int) (*Tensor, error) {
	if len(axes) == 0 {
		// when axes is empty, sum all.
		var result float64
		iter := t.Iterator()
		for iter.HasNext() {
			result += iter.Next()
		}

		if keepdims {
			return NdShape([]float64{result}, all(1, len(t.Shape))...)
		}

		return Scalar(result), nil
	}

	slices.Sort(axes)

	// check axes. They must be unique && 0 < axes < ndim - 1.

	copied := slices.Compact(copySlice(axes))
	if len(copied) != len(axes) {
		return nil, fmt.Errorf("duplicate value in axes: %v", axes)
	}

	if slices.ContainsFunc(copied, func(axis int) bool {
		return axis < 0 || t.Ndim()-1 < axis
	}) {
		return nil, fmt.Errorf("axis out of bounds: %v", axes)
	}

	// check okay, do sum.

	// total size of summed tensor
	length := t.Size()

	// summed tensor shape
	newshape := copyShape(t.Shape)

	filtered := make([]int, len(axes))

	for i, axis := range axes {
		length /= t.Shape[axis]
		newshape[axis] = 1
		filtered[i] = t.Shape[axis]
	}

	c := cartesian(filtered)
	// accessing slices in sum
	slices := make([][]*Slice, length)

	for i := range t.Ndim() {
		if !slices.Contains(axes, i) {
			
		}
	}


	indices := cartesian(pick)
	for _, index := range indices {
		t.Slices()
	}


	return nil, nil
}
