package tensor2

import (
	"fmt"
)

func (t *Tensor) matrixRow(row int) []float64 {
	result := make([]float64, t.Shape[1])
	for i := range result {
		result[i] = t.data[t.offset+row*t.Strides[0]+i*t.Strides[1]]
	}
	return result
}

func (t *Tensor) matrixCol(col int) []float64 {
	result := make([]float64, t.Shape[0])
	for i := range result {
		result[i] = t.data[t.offset+col*t.Strides[1]+i*t.Strides[0]]
	}
	return result
}

func (er *tensorErrResponser) Dot(t2 *Tensor) (*Tensor, error) {
	if er.t.Ndim() != 2 || t2.Ndim() != 2 {
		return nil, fmt.Errorf("Dot() requires matrix x matrix but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	if er.t.Shape[1] != t2.Shape[0] {
		return nil, fmt.Errorf("Dot() requires shape1[1] is equal to shape2[0], but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	rownum, colnum := er.t.Shape[0], t2.Shape[1]

	newshape := []int{rownum, colnum}

	data := make([]float64, rownum*colnum)

	rows := make([][]float64, rownum)
	for r := range rownum {
		rows[r] = er.t.matrixRow(r)
	}

	cols := make([][]float64, colnum)
	for c := range colnum {
		cols[c] = t2.matrixCol(c)
	}

	calcsize := len(rows[0])

	type calcresult struct {
		row  int
		vals []float64
	}

	ch := make(chan calcresult, rownum)
	for r := range rows {
		go func(rn int) {
			results := make([]float64, colnum)
			for c := range cols {
				var result float64
				for j := range calcsize {
					result += rows[rn][j] * cols[c][j]
				}
				results[c] = result

			}
			ch <- calcresult{row: rn, vals: results}
		}(r)
	}

	for _ = range len(rows) {
		result := <-ch
		for i := range result.vals {
			data[result.row*colnum+i] = result.vals[i]
		}
	}

	return RespErr.NdShape(data, newshape...)
}

func (t *Tensor) Dot(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Dot(t2))
}
