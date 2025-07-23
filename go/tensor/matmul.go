package tensor

import (
	"fmt"

	"github.com/hidetatz/whale/blas"
)

func (er *tensorErrResponser) Matmul(t2 *Tensor) (*Tensor, error) {
	if er.t.Ndim() != 2 || t2.Ndim() != 2 {
		return nil, fmt.Errorf("Dot() requires matrix x matrix but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	if er.t.Shape[1] != t2.Shape[0] {
		return nil, fmt.Errorf("Dot() requires shape1[1] is equal to shape2[0], but got shape %v x %v", er.t.Shape, t2.Shape)
	}

	at := blas.NoTrans
	if er.t.Strides[0] < er.t.Strides[1] {
		at = blas.Trans
	}

	bt := blas.NoTrans
	if t2.Strides[0] < t2.Strides[1] {
		bt = blas.Trans
	}

	lda := er.t.Strides[0]
	if at == blas.Trans {
		lda = er.t.Strides[1]
	}

	ldb := t2.Strides[0]
	if bt == blas.Trans {
		ldb = t2.Strides[1]
	}

	c := make([]float32, er.t.Shape[0]*t2.Shape[1])
	blas.Sgemm(at, bt, er.t.Shape[0], t2.Shape[1], er.t.Shape[1], 1, er.t.data[er.t.offset:], lda, t2.data[t2.offset:], ldb, 0, c, t2.Shape[1])

	return NdShape(c, er.t.Shape[0], t2.Shape[1]), nil
}

func (t *Tensor) Matmul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Matmul(t2))
}
