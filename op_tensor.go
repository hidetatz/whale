package whale

import (
	"fmt"
	"slices"
)

/*
 * Tensor modification
 */

// Reshape reshapes the given tensor to the specified shape.
func Reshape(v *Variable, shape ...int) (*Variable, error) {
	if slices.Equal(v.data.Shape(), shape) {
		return v, nil
	}

	f := NewFunction(&reshape{origshape: v.data.CopyShape(), shape: shape})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type reshape struct {
	origshape []int
	shape     []int
}

func (r *reshape) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.Reshape(r.shape...)
	if err != nil {
		return nil, fmt.Errorf("reshape: %w", err)
	}

	return asvars(y), nil
}

func (r *reshape) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := Reshape(gy[0], r.origshape...)
	if err != nil {
		return nil, fmt.Errorf("reshape backward: %w", err)
	}
	return []*Variable{y}, nil
}

func (r *reshape) String() string { return "reshape" }

// Transpose transposes the tensor.
func Transpose(v *Variable) (*Variable, error) {
	f := NewFunction(&transpose{})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type transpose struct{}

func (t *transpose) Forward(inputs ...*Variable) ([]*Variable, error) {
	return asvars(inputs[0].data.Transpose()), nil
}

func (t *transpose) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := Transpose(gy[0])
	if err != nil {
		return nil, err
	}

	return []*Variable{y}, nil
}

func (t *transpose) String() string {
	return "T"
}

// BroadcastTo broadcasts the tensor to the given shape.
func BroadcastTo(v *Variable, shape ...int) (*Variable, error) {
	if slices.Equal(v.data.CopyShape(), shape) {
		return v, nil
	}

	f := NewFunction(&broadcastTo{origshape: v.data.CopyShape(), shape: shape})
	y, err := f.forward(v)
	if err != nil {
		return nil, err
	}

	return y[0], nil
}

type broadcastTo struct {
	origshape []int
	shape     []int
}

func (b *broadcastTo) Forward(inputs ...*Variable) ([]*Variable, error) {
	y, err := inputs[0].data.BroadcastTo(b.shape...)
	if err != nil {
		return nil, fmt.Errorf("BroadcastTo: %w", err)
	}

	return asvars(y), nil
}

func (b *broadcastTo) Backward(gy ...*Variable) ([]*Variable, error) {
	y, err := SumTo(gy[0], b.origshape...)
	if err != nil {
		return nil, fmt.Errorf("BroadcastTo Backward: %w", err)
	}
	return []*Variable{y}, nil
}

func (b *broadcastTo) String() string {
	return "BroadcastTo"
}
