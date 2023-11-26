package whale

import (
	"fmt"
	"strings"
)

type Tensor struct {
	val   []float64
	shape []int
	dim   int
}

func TensorByScalar(data float64) *Tensor {
	return &Tensor{val: []float64{data}}
}

func TensorNd(data []float64, shape ...int) *Tensor {
	return &Tensor{val: data, shape: shape, dim: len(shape)}
}

func Zeros(shape ...int) *Tensor {
	datalen := 1
	for _, s := range shape {
		datalen *= s
	}
	data := make([]float64, datalen)
	return TensorNd(data, shape...)
}

func Ones(shape ...int) *Tensor {
	datalen := 1
	for _, s := range shape {
		datalen *= s
	}
	data := make([]float64, datalen)
	for i := range data {
		data[i] = 1
	}
	return TensorNd(data, shape...)
}

func (t *Tensor) String() string {
	s := ""

	// declaration required to be called recursively
	var printTensorRec func(data []float64, shape []int, depth int, indent string)

	printTensorRec = func(data []float64, shape []int, depth int, indent string) {
		if len(shape) == 0 {
			s += fmt.Sprintf("%s%v\n", indent, data)
			return
		}

		if len(shape) == 1 {
			s += fmt.Sprintf("%s%v\n", indent, data)
			return
		}

		size := shape[0]
		for i := 0; i < size; i++ {
			newIndent := strings.Repeat("  ", depth)
			s += fmt.Sprintf("%s[\n", indent)
			printTensorRec(data[i*size:(i+1)*size], shape[1:], depth+1, newIndent)
			s += fmt.Sprintf("%s]\n", indent)
		}
	}

	printTensorRec(t.val, t.shape, 1, "")
	return s
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Dim() int {
	return t.dim
}
