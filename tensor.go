package whale

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

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Dim() int {
	return t.dim
}
