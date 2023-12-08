package whale

import (
	"math"

	"github.com/hidetatz/whale/tensor"
)

type CPU struct{}

func (c *CPU) Pow(t, y *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = math.Pow(t.Data[i], y.Data[i])
	}
	return result
}

func (c *CPU) Exp(t *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = math.Exp(t.Data[i])
	}
	return result
}

func (c *CPU) Add(t1, t2 *tensor.Tensor) *tensor.Tensor {
	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] + t2.Data[i]
	}
	return result
}

func (c *CPU) Sub(t1, t2 *tensor.Tensor) *tensor.Tensor {
	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] - t2.Data[i]
	}
	return result
}

func (c *CPU) Mul(t1, t2 *tensor.Tensor) *tensor.Tensor {
	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] * t2.Data[i]
	}
	return result
}

func (c *CPU) Div(t1, t2 *tensor.Tensor) *tensor.Tensor {
	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] / t2.Data[i]
	}
	return result
}

func (c *CPU) Neg(t *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = -t.Data[i]
	}
	return result
}

func (c *CPU) Sin(t *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = math.Sin(t.Data[i])
	}
	return result
}

func (c *CPU) Cos(t *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = math.Cos(t.Data[i])
	}
	return result
}

func (c *CPU) Tanh(t *tensor.Tensor) *tensor.Tensor {
	result := t.Copy()
	for i := range result.Data {
		result.Data[i] = math.Tanh(t.Data[i])
	}
	return result
}
