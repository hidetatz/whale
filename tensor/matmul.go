//go:build !gonum

package tensor

import (
	"fmt"
	"sync"
)

func (t *Tensor) matrixRow(row int) []float32 {
	result := make([]float32, t.Shape[1])
	for i := range result {
		result[i] = t.data[t.offset+row*t.Strides[0]+i*t.Strides[1]]
	}
	return result
}

func (t *Tensor) matrixCol(col int) []float32 {
	result := make([]float32, t.Shape[0])
	for i := range result {
		result[i] = t.data[t.offset+col*t.Strides[1]+i*t.Strides[0]]
	}
	return result
}

func (er *tensorErrResponser) Matmul(t2 *Tensor) (*Tensor, error) {
	if er.t.Ndim() != 2 || t2.Ndim() != 2 {
		return nil, fmt.Errorf("Dot() requires matrix x matrix but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	if er.t.Shape[1] != t2.Shape[0] {
		return nil, fmt.Errorf("Dot() requires shape1[1] is equal to shape2[0], but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	// return Matmul_v1(er.t, t2), nil
	// return Matmul_v2(er.t, t2), nil
	return Matmul_v3(er.t, t2), nil
}

func (t *Tensor) Matmul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Matmul(t2))
}

// parallelize by row and col, significantly slow
func Matmul_v3(t1, t2 *Tensor) *Tensor {
	rownum, colnum := t1.Shape[0], t2.Shape[1]
	newshape := []int{rownum, colnum}
	data := make([]float32, rownum*colnum)
	rows := make([][]float32, rownum)
	for r := range rownum {
		rows[r] = t1.matrixRow(r)
	}

	cols := make([][]float32, colnum)
	for c := range colnum {
		cols[c] = t2.matrixCol(c)
	}

	wg := sync.WaitGroup{}
	for r := range rows {
		for c := range cols {
			wg.Add(1)
			go func(r, c int) {
				var result float32
				for i := range t1.Shape[1] {
					result += rows[r][i] * cols[c][i]
				}
				data[r*colnum+c] = result
				wg.Done()
			}(r, c)
		}
	}

	wg.Wait()
	return NdShape(data, newshape...)
}

// parallelize by row, no use channel but copy computation result directly into `data`
func Matmul_v2(t1, t2 *Tensor) *Tensor {
	rownum, colnum := t1.Shape[0], t2.Shape[1]
	newshape := []int{rownum, colnum}
	data := make([]float32, rownum*colnum)
	rows := make([][]float32, rownum)
	for r := range rownum {
		rows[r] = t1.matrixRow(r)
	}

	cols := make([][]float32, colnum)
	for c := range colnum {
		cols[c] = t2.matrixCol(c)
	}

	wg := sync.WaitGroup{}
	for r := range rows {
		wg.Add(1)
		go func(rn int) {
			results := make([]float32, colnum)
			for c := range cols {
				var result float32
				for j := range t1.Shape[1] {
					result += rows[rn][j] * cols[c][j]
				}
				results[c] = result

			}
			copy(data[rn*colnum:rn*colnum+len(results)], results)
			wg.Done()
		}(r)
	}

	wg.Wait()
	return NdShape(data, newshape...)
}

// parallelize by row
func Matmul_v1(t1, t2 *Tensor) *Tensor {
	rownum, colnum := t1.Shape[0], t2.Shape[1]
	newshape := []int{rownum, colnum}
	data := make([]float32, rownum*colnum)
	rows := make([][]float32, rownum)
	for r := range rownum {
		rows[r] = t1.matrixRow(r)
	}

	cols := make([][]float32, colnum)
	for c := range colnum {
		cols[c] = t2.matrixCol(c)
	}

	calcsize := len(rows[0])

	type calcresult struct {
		row  int
		vals []float32
	}

	ch := make(chan calcresult, rownum)
	for r := range rows {
		go func(rn int) {
			results := make([]float32, colnum)
			for c := range cols {
				var result float32
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

	return NdShape(data, newshape...)
}
