package tensor

import "slices"

func (t *Tensor) Min() *Tensor {
	return Scalar(slices.Min(t.Ravel()))
}

func (t *Tensor) Max() *Tensor {
	return Scalar(slices.Max(t.Ravel()))
}
