package tensor

import (
	"cmp"
	"fmt"
	"slices"
)

func (t *Tensor) view(offset int, shape, strides []int) *Tensor {
	return &Tensor{data: t.data, offset: offset, Shape: shape, Strides: strides, isview: true}
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	return MustGet(t.ErrResponser().Reshape(shape...))
}

func (er *tensorErrResponser) Reshape(shape ...int) (*Tensor, error) {
	if product(shape) != er.t.Size() {
		return nil, fmt.Errorf("cannot reshape size %v tensor into %v", er.t.Size(), shape)
	}

	strides := make([]int, len(shape))
	for i := range shape {
		strides[i] = product(shape[i+1:])
	}

	// reshape shares original tensor data/offset, only shape and strides are modified.
	return er.t.view(er.t.offset, shape, strides), nil
}

func (t *Tensor) Transpose(axes ...int) *Tensor {
	return MustGet(t.ErrResponser().Transpose(axes...))
}

func (er *tensorErrResponser) Transpose(axes ...int) (*Tensor, error) {
	if er.t.IsScalar() {
		return er.t, nil
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
		axes = seq(0, er.t.Ndim())
		slices.Reverse(axes)
	} else {
		// else, axes must be an arbitrarily sorted slice of [0, 1, 2, ... dim].
		copied := copySlice(axes)
		slices.Sort(copied)
		if !slices.Equal(copied, seq(0, er.t.Ndim())) {
			return nil, fmt.Errorf("axes don't much: %v for shape %v", axes, er.t.Shape)
		}
	}

	// do transpose

	t2 := er.t.view(er.t.offset, make([]int, er.t.Ndim()), make([]int, er.t.Ndim()))
	for i, axis := range axes {
		t2.Shape[i] = er.t.Shape[axis]
		t2.Strides[i] = er.t.Strides[axis]
	}

	return t2, nil
}

func (t *Tensor) Squeeze(axes ...int) *Tensor {
	return MustGet(t.ErrResponser().Squeeze(axes...))
}

func (er *tensorErrResponser) Squeeze(axes ...int) (*Tensor, error) {
	for _, axis := range axes {
		if er.t.Ndim() < axis {
			return nil, fmt.Errorf("axis out of bounds: %v for %v tensor", axis, er.t.Ndim())
		}

		if er.t.Shape[axis] != 1 {
			return nil, fmt.Errorf("non-1 axis is specified: %v for axis whose size is %v", axis, er.t.Shape[axis])
		}
	}

	newshape := []int{}
	newstrides := []int{}
	for i := range er.t.Shape {
		if er.t.Shape[i] != 1 {
			newshape = append(newshape, er.t.Shape[i])
			newstrides = append(newstrides, er.t.Strides[i])
			continue
		}

		if len(axes) != 0 && !slices.Contains(axes, i) {
			newshape = append(newshape, er.t.Shape[i])
			newstrides = append(newstrides, er.t.Strides[i])
		}
	}

	return er.t.view(er.t.offset, newshape, newstrides), nil
}

func (_ *plainErrResponser) Broadcast(t1, t2 *Tensor) (newt1, newt2 *Tensor, err error) {
	broadcastedShape, err := CanBroadcast([]*Tensor{t1, t2})

	if err != nil {
		return nil, nil, err
	}

	nt1, err := t1.ErrResponser().BroadcastTo(broadcastedShape...)
	if err != nil {
		return nil, nil, err
	}

	nt2, err := t2.ErrResponser().BroadcastTo(broadcastedShape...)
	if err != nil {
		return nil, nil, err
	}

	return nt1, nt2, nil
}

func Broadcast(t1, t2 *Tensor) (newt1, newt2 *Tensor) {
	return MustGet2(RespErr.Broadcast(t1, t2))
}

func CanBroadcast(tensors []*Tensor) ([]int, error) {
	// just for error message...
	origshapes := make([][]int, len(tensors))

	shapes := make([][]int, len(tensors))
	for i, t := range tensors {
		origshapes[i] = copySlice(t.Shape)
		shapes[i] = copySlice(t.Shape)
	}

	longest := slices.MaxFunc(shapes, func(a, b []int) int { return cmp.Compare(len(a), len(b)) })

	// unify the shape length by pushing 1 front
	for i, shape := range shapes {
		shapes[i] = slices.Concat(all(1, len(longest)-len(shape)), shape)
	}

	newshape := make([]int, len(longest))
	for i := 0; i < len(longest); i++ {
		// lengths will contains i-dimension length for each shapes
		lengths := make([]int, len(shapes))
		for j, shape := range shapes {
			lengths[j] = shape[i]
		}

		l := 0
		for _, length := range lengths {
			if length == 1 {
				continue
			}

			if l == 0 {
				l = length
				continue
			}

			if l != length {
				return nil, fmt.Errorf("cannot broadcast tensors with shapes: %v", origshapes)
			}
		}

		if l == 0 {
			// when all length is 1, comes here.
			l = 1
		}

		// i-dim is broadcastable.
		newshape[i] = l
	}

	return newshape, nil
}

func (t *Tensor) BroadcastTo(shape ...int) *Tensor {
	return MustGet(t.ErrResponser().BroadcastTo(shape...))
}

func (er *tensorErrResponser) BroadcastTo(shape ...int) (*Tensor, error) {
	if len(er.t.Shape) > len(shape) {
		return nil, fmt.Errorf("invalid desired shape")
	}

	delta := len(shape) - len(er.t.Shape)
	newstrides := make([]int, len(shape))
	for i := er.t.Ndim() - 1; 0 <= i; i-- {
		if shape[delta+i] == er.t.Shape[i] && er.t.Shape[i] == 1 {
			newstrides[delta+i] = 0
			continue
		}

		if shape[delta+i] == er.t.Shape[i] && er.t.Shape[i] != 1 {
			newstrides[delta+i] = er.t.Strides[i]
			continue
		}

		if er.t.Shape[i] != 1 {
			return nil, fmt.Errorf("cannot broadcast: original shape is %v, target is %v", er.t.Shape, shape)
		}

		newstrides[delta+i] = 0
	}

	return er.t.view(er.t.offset, shape, newstrides), nil
}
