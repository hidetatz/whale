package whale

import (
	"math"
	"math/rand"

	"github.com/hidetatz/whale/tensor"
)

func RandSin(shape ...int) (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.Rand(shape...)

	sin := tensor.Scalar(2 * math.Pi).Mul(x)
	y := tensor.Rand(shape...).Add(sin)
	return x, y
}

func RandSpiral() (*tensor.Tensor, *tensor.Tensor) {
	x := [][]float32{}
	t := []float32{}
	for j := 0; j < 3; j++ {
		for i := range 100 {
			var rate float32 = float32(i) / 100.0
			var radius float32 = 1.0 * rate
			theta := float32(j)*4 + 4*rate + float32(rand.NormFloat64())*0.2
			x = append(x, []float32{radius * float32(math.Sin(float64(theta))), radius * float32(math.Cos(float64(theta)))})
			t = append(t, float32(j))
		}
	}

	//todo: shuffle x and t

	flat := []float32{}
	for i := range x {
		flat = append(flat, x[i]...)
	}

	xt := tensor.NdShape(flat, 300, 2)
	tt := tensor.Vector(t)

	return xt, tt
}
