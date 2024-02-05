package whale

import (
	"math"
	"math/rand"

	"github.com/hidetatz/whale/tensor"
)

func RandSin(shape ...int) (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.Rand(shape...)
	y := device.Add(device.Sin(device.Mul(x, device.Mul(tensor.Scalar(2), tensor.Scalar(math.Pi)))), tensor.Rand(shape...))
	return x, y
}

func RandSpiral() (*tensor.Tensor, *tensor.Tensor) {
	x := [][]float64{}
	t := []float64{}
	for j := 0; j < 3; j++ {
		for i := 0; i < 100; i++ {
			var rate float64 = float64(i) / 100.0
			var radius float64 = 1.0 * rate
			theta := float64(j)*4 + 4*rate + rand.NormFloat64()*0.2
			x = append(x, []float64{radius * math.Sin(theta), radius * math.Cos(theta)})
			t = append(t, float64(j))
		}
	}

	//todo: shuffle x and t

	flat := []float64{}
	for i := range x {
		flat = append(flat, x[i]...)
	}

	xt := tensor.MustNd(flat, 300, 2)
	tt := tensor.Vector(t)

	return xt, tt
}