package tensor2

import "slices"

func (t *Tensor) Min() *Tensor {
	return Scalar(slices.Min(t.Flatten()))
}

func (t *Tensor) Max() *Tensor {
	return Scalar(slices.Max(t.Flatten()))
}
