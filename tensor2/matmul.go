package tensor2

import "fmt"

func (t *Tensor) matrixRow(row int) ([]float64, error) {
	if t.Ndim() != 2 {
		return nil, fmt.Errorf("row is not defined on non-matrix")
	}

	result := make([]float64, t.Shape[1])
	for i := range result {
		result[i] = t.data[t.offset+row*t.Strides[0]+i*t.Strides[1]]
	}

	return result, nil
}

func (t *Tensor) matrixCol(col int) ([]float64, error) {
	if t.Ndim() != 2 {
		return nil, fmt.Errorf("col is not defined on non-matrix")
	}

	result := make([]float64, t.Shape[0])
	for i := range result {
		result[i] = t.data[t.offset+col*t.Strides[1]+i*t.Strides[0]]
	}

	return result, nil
}

func (t *Tensor) Dot(t2 *Tensor) (*Tensor, error) {
	if t.Ndim() != 2 || t2.Ndim() != 2 {
		return nil, fmt.Errorf("Dot() requires matrix x matrix but got shape %v x %v", t.Shape, t2.Shape)
	}

	if t.Shape[1] != t2.Shape[0] {
		return nil, fmt.Errorf("Dot() requires shape1[1] is equal to shape2[0], but got shape %v x %v", t.Shape, t2.Shape)
	}

	rownum, colnum := t.Shape[0], t2.Shape[1]

	newshape := []int{rownum, colnum}

	data := make([]float64, rownum*colnum)
	i := 0
	for r := range rownum {
		row, err := t.matrixRow(r)
		if err != nil {
			return nil, err
		}

		for c := range colnum {
			col, err := t2.matrixCol(c)
			if err != nil {
				return nil, err
			}

			var result float64
			for j := range row {
				result += row[j] * col[j]
			}
			data[i] = result
			i++
		}
	}

	return RespErr.NdShape(data, newshape...)
}
