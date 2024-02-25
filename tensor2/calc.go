package tensor2

import (
	"fmt"
	"slices"
)

func (t *Tensor) Sum(keepdims bool, axes ...int) (*Tensor, error) {
	if t.IsScalar() {
		if len(axes) != 0 {
			return nil, fmt.Errorf("axis out of bounds: %v", axes)
		}
		return t, nil
	}

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

	// create slices. Let's say shape is [2, 3, 4], axes is [0, 2],
	// result will be
	// [[0, :, 0], [0, :, 1], [0, :, 2], [0, :, 3], [1, :, 0], [1, :, 1], [1, :, 2], [1, :, 3]]
	ss := [][]*Slice{}
	template := make([]*Slice, t.Ndim())
	for i := range template {
		template[i] = All()
	}

	var gen func(int, []*Slice)
	gen = func(index int, current []*Slice) {
		if index == len(axes) {
			ss = append(ss, append([]*Slice(nil), current...))
			return
		}

		for i := 0; i < t.Shape[axes[index]]; i++ {
			current[axes[index]] = At(i)
			gen(index+1, current)
			current[axes[index]] = All()
		}
	}
	gen(0, template)

	// extract each slice and sum them
	data := make([]float64, t.Size()/len(ss))
	for _, s := range ss {
		t2, err := t.Slice(s...)
		if err != nil {
			panic(err)
		}

		flat := t2.Flatten()
		for i := range len(data) {
			data[i] += flat[i]
		}
	}

	newshape := copySlice(t.Shape)
	if keepdims {
		// if keepdims, dim will be 1
		for _, axis := range axes {
			newshape[axis] = 1
		}
	} else {
		// else, the dim is reduced
		slices.Reverse(axes)
		for _, axis := range axes {
			newshape = append(newshape[:axis], newshape[axis+1:]...)
		}
	}

	return NdShape(data, newshape...)
}

func (t *Tensor) SumTo(shape ...int) (*Tensor, error) {
	ndim := len(shape)
	lead := t.Ndim() - ndim
	leadAxis := seq[int](0, lead)

	var axes []int
	for i, dim := range shape {
		if dim == 1 {
			axes = append(axes, i+lead)
		}
	}

	y, err := t.Sum(true, append(leadAxis, axes...)...)
	if err != nil {
		return nil, err
	}

	if lead > 0 {
		y2, err := y.Squeeze(leadAxis...)
		if err != nil {
			return nil, err
		}
		y = y2
	}

	return y, nil
}
