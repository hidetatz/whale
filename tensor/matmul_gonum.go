//go:build gonum

package tensor

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

var b = blas32.Implementation()

func (er *tensorErrResponser) Matmul(t2 *Tensor) (*Tensor, error) {
	m, k, n := er.t.Shape[0], er.t.Shape[1], t2.Shape[1]
	alpha := float32(1.0)
	A := er.t.Flatten()
	B := t2.Flatten()
	beta := float32(0.0)
	C := make([]float32, m*n)
	b.Sgemm(blas.NoTrans, blas.NoTrans, m, n, k, alpha, A, k, B, n, beta, C, n)
	return RespErr.NdShape(C, m, n)
}

func (t *Tensor) Matmul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Matmul(t2))
}
