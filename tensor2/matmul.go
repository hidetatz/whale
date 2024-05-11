package tensor2

import "fmt"

func (t *Tensor) Dot(t2 *Tensor) (*Tensor, error) {
	if t.Ndim() != 2 || t2.Ndim() != 2 {
		return nil, fmt.Errorf("Dot() requires matrix x matrix but got shape %v x %v", t.Shape, t2.Shape)
	}

	if t.Shape[1] != t2.Shape[0] {
		return nil, fmt.Errorf("Dot() requires shape1[1] is equal to shape2[0], but got shape %v x %v", t.Shape, t2.Shape)
	}

	rows, cols := t.Shape[0], t2.Shape[1]

	newshape := []int{rows, cols}

	data := make([]float64, rows*cols)
	i := 0
	for row := range rows {
		for col := range cols {
			t1row, err := t.Index(At(row), All())
			if err != nil {
				return nil, err
			}

			t2col, err := t2.Index(All(), At(col))
			if err != nil {
				return nil, err
			}

			t1iter := t1row.Iterator()
			t2iter := t2col.Iterator()
			var result float64
			for t1iter.HasNext() {
				_, v1 := t1iter.Next()
				_, v2 := t2iter.Next()
				result += v1 * v2
			}

			data[i] = result
			i++
		}
	}

	return NdShape(data, newshape...)
}
