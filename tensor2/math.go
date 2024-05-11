package tensor2

import "fmt"

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
	i := 0
	for iter.HasNext() {
		f := iter.Next()
		if i == 0 {
			cur = f
			arg = 0
			i++
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
		i++
	}

	return arg
}
