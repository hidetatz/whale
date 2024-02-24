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

	newshape := make([]int, t.Ndim()-len(axes))
	for i, dim := range t.Shape {
		if !slices.Contains(axes, i) {
			newshape = append(newshape, dim)
		}
	}

	length := product(newshape)

	tensors := make([]*Tensor, length)

	return nil, nil
}
