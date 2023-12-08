package whale

import "github.com/hidetatz/whale/tensor"

type Device interface {
	Pow(t, c *tensor.Tensor) *tensor.Tensor
	Exp(t *tensor.Tensor) *tensor.Tensor
	Add(t1, t2 *tensor.Tensor) *tensor.Tensor
	Sub(t1, t2 *tensor.Tensor) *tensor.Tensor
	Mul(t1, t2 *tensor.Tensor) *tensor.Tensor
	Div(t1, t2 *tensor.Tensor) *tensor.Tensor
	Neg(t *tensor.Tensor) *tensor.Tensor
	Sin(t *tensor.Tensor) *tensor.Tensor
	Cos(t *tensor.Tensor) *tensor.Tensor
	Tanh(t *tensor.Tensor) *tensor.Tensor
}
