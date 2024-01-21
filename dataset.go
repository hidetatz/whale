package whale

import (
	"math"

	"github.com/hidetatz/whale/tensor"
)

func RandSin(shape ...int) (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.Rand(shape...)
	y := device.Add(device.Sin(device.Mul(x, device.Mul(tensor.Scalar(2), tensor.Scalar(math.Pi)))), tensor.Rand(shape...))
	return x, y
}
