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

func (c *CPU) MatMul(t1, t2 *tensor.Tensor) *tensor.Tensor {
	tc1 := t1.Copy()
	tc2 := t2.Copy()

	if tc1.IsVector() && tc2.IsVector() {
		for i := range tc1.Data {
			tc1.Data[i] = tc1.Data[i] * tc2.Data[i]
		}
		return tc1
	}

	if tc1.Dim() == 2 && tc2.Dim() == 2 {
		shape1 := tc1.CopyShape()
		shape2 := tc2.CopyShape()
		if shape1[1] != shape2[0] {
			panic("matmul failed: invalid shape")
		}

		tomatrix := func(t *tensor.Tensor) [][]float64 {
			shape := t.CopyShape()
			col, row := shape[0], shape[1]
			result := [][]float64{}
			for i := 0; i < col; i++ {
				result = append(result, t.Data[i*row:i*row+row])
			}
			return result

		}

		matrix1 := tomatrix(tc1)
		matrix2 := tomatrix(tc2)

		targetShape := []int{shape1[0], shape2[1]}

		result := matmul(matrix1, matrix2)
		data := flatten(result)
		t, _ := tensor.Nd(data, targetShape...)
		return t
	}

	panic("matmul is possible only for vector x vector or 2d x 2d")
}

func flatten(matrix [][]float64) []float64 {
	result := []float64{}
	for _, m := range matrix {
		result = append(result, m...)
	}
	return result
}

func matmul(matrixA, matrixB [][]float64) [][]float64 {
	rowsA, colsA := len(matrixA), len(matrixA[0])
	rowsB, colsB := len(matrixB), len(matrixB[0])

	if colsA != rowsB {
		panic("Invalid matrix dimensions for multiplication")
	}

	result := make([][]float64, rowsA)
	for i := range result {
		result[i] = make([]float64, colsB)
	}

	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsB; j++ {
			for k := 0; k < colsA; k++ {
				result[i][j] += matrixA[i][k] * matrixB[k][j]
			}
		}
	}

	return result
}
