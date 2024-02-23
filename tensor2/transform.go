package tensor2

import "fmt"

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
