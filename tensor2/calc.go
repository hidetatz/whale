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
			_, v := iter.Next()
			result += v
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
	ss := [][]*IndexArg{}
	template := make([]*IndexArg, t.Ndim())
	for i := range template {
		template[i] = All()
	}

	var gen func(int, []*IndexArg)
	gen = func(index int, current []*IndexArg) {
		if index == len(axes) {
			ss = append(ss, append([]*IndexArg(nil), current...))
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
		t2, err := t.Index(s...)
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

func (t *Tensor) Argmax(keepdims bool, axis int) (*Tensor, error) {
	return t.argFunc(keepdims, axis, "max")
}

func (t *Tensor) Argmin(keepdims bool, axis int) (*Tensor, error) {
	return t.argFunc(keepdims, axis, "min")
}

func (t *Tensor) argFunc(keepdims bool, axis int, fn string) (*Tensor, error) {
	if fn != "max" && fn != "min" {
		// this must not happen
		panic("argFunc received invalid fn value: " + fn)
	}

	if t.Ndim() <= axis {
		return nil, fmt.Errorf("axis %v	is out of bounds for array dimension is %v", axis, t.Ndim())
	}

	if axis < 0 {
		arg := t.argFuncFlat(fn)

		if !keepdims {
			return Scalar(float64(arg)), nil
		}

		return NdShape([]float64{float64(arg)}, all(1, t.Ndim())...)
	}

	newshape := copySlice(t.Shape)
	if keepdims {
		newshape[axis] = 1
	} else {
		newshape = append(newshape[:axis], newshape[axis+1:]...)
	}

	data := make([]float64, product(newshape))
	shp := copySlice(t.Shape)
	shp[axis] = 1

	indexArgs := cartesianIdx(shp)
	for i, indexArg := range indexArgs {
		indexArg[axis] = All()
		t2, err := t.Index(indexArg...)
		if err != nil {
			// this must not happen
			panic("index() returns err: " + err.Error())
		}

		arg := t2.argFuncFlat(fn)
		data[i] = float64(arg)
	}

	return NdShape(data, newshape...)
}

func (t *Tensor) argFuncFlat(fn string) int {
	var cur float64 // actual value
	var arg int     // index to be returned
	iter := t.Iterator()
	for iter.HasNext() {
		i, f := iter.Next()
		if i == 0 {
			cur = f
			arg = 0
			continue
		}

		update := false
		if fn == "max" {
			update = cur < f
		} else {
			update = f < cur
		}

		if update {
			cur = f
			arg = i
		}
	}

	return arg
}

func (t *Tensor) Clip(min, max float64) *Tensor {
	data := make([]float64, t.Size())

	iter := t.Iterator()
	for iter.HasNext() {
		i, f := iter.Next()
		if f < min {
			f = min
		}

		if max < f {
			f = max
		}

		data[i] = f
	}

	return Must(NdShape(data, copySlice(t.Shape)...))
}
