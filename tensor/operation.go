package tensor

import (
	"github.com/hidetatz/whale/cuda"
)

var cudaAvail = cuda.IsAvailable()

func Pow(t *Tensor, y float64) *Tensor {
	if cudaAvail {
		return PowCUDA(t, y)
	}

	return PowCPU(t, y)
}

func Exp(t *Tensor) *Tensor {
	if cudaAvail {
		return ExpCUDA(t)
	}

	return ExpCPU(t)
}

func PowCUDA(t *Tensor, y float64) *Tensor {
	panic("not implemented!")
}

func ExpCUDA(t *Tensor) *Tensor {
	panic("not implemented!")
}
