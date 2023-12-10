package whale

import (
	"fmt"
	"math"

	"github.com/hidetatz/whale/tensor"
)

type CPU struct{}

func uniformShape(t1, t2 *tensor.Tensor) (newt1, newt2 *tensor.Tensor, err error) {
	shape1 := t1.CopyShape()
	shape2 := t2.CopyShape()

	// First, push 1 to head until the length gets the same
	for len(shape1) != len(shape2) {
		if len(shape1) > len(shape2) {
			shape2 = append([]int{1}, shape2...)
		} else {
			shape1 = append([]int{1}, shape1...)
		}
	}

	// Second, check the values are the same one of the value is 1 for each shape value.
	// If not, Broadcasting is not possible.
	for i := range shape1 {
		if shape1[i] == shape2[i] {
			continue
		}

		if shape1[i] == 1 {
			// shape1 will be the shape for the broadcasting target
			shape1[i] = shape2[i]
			continue
		}

		if shape2[i] == 1 {
			continue
		}

		return nil, nil, fmt.Errorf("broadcasting is impossible for the shape: %v and %v", t1.Shape(), t2.Shape())
	}

	nt1, err := t1.BroadcastTo(shape1...)
	if err != nil {
		return nil, nil, fmt.Errorf("broadcasting failed: %v", err)
	}

	nt2, err := t2.BroadcastTo(shape1...)
	if err != nil {
		return nil, nil, fmt.Errorf("broadcasting failed: %v", err)
	}

	return nt1, nt2, nil
}

func (c *CPU) Pow(t1, t2 *tensor.Tensor) *tensor.Tensor {
	t1, t2, err := uniformShape(t1, t2)
	if err != nil {
		panic(err)
	}

	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = math.Pow(t1.Data[i], t2.Data[i])
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
	t1, t2, err := uniformShape(t1, t2)
	if err != nil {
		panic(err)
	}

	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] + t2.Data[i]
	}
	return result
}

func (c *CPU) Sub(t1, t2 *tensor.Tensor) *tensor.Tensor {
	t1, t2, err := uniformShape(t1, t2)
	if err != nil {
		panic(err)
	}

	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] - t2.Data[i]
	}
	return result
}

func (c *CPU) Mul(t1, t2 *tensor.Tensor) *tensor.Tensor {
	t1, t2, err := uniformShape(t1, t2)
	if err != nil {
		panic(err)
	}

	result := t1.Copy()
	for i := range result.Data {
		result.Data[i] = t1.Data[i] * t2.Data[i]
	}
	return result
}

func (c *CPU) Div(t1, t2 *tensor.Tensor) *tensor.Tensor {
	t1, t2, err := uniformShape(t1, t2)
	if err != nil {
		panic(err)
	}

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
