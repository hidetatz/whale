//go:build blas

package tensor

import (
	/*
	   #cgo LDFLAGS: -L/opt/OpenBLAS/lib/ -lopenblas
	   #cgo CFLAGS: -I /opt/OpenBLAS/include/
	   #include <cblas.h>
	*/
	"C"
	"unsafe"
)

func (er *tensorErrResponser) Matmul(t2 *Tensor) (*Tensor, error) {
	m := C.blasint(er.t.Shape[0])
	k := C.blasint(er.t.Shape[1])
	n := C.blasint(t2.Shape[1])

	alpha := C.float(1)
	beta := C.float(0)

	c := make([]float32, er.t.Shape[0]*t2.Shape[1])

	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		m,
		n,
		k,
		alpha,
		(*C.float)(unsafe.Pointer(&er.t.Ravel()[0])),
		k,
		(*C.float)(unsafe.Pointer(&t2.Ravel()[0])),
		n,
		beta,
		(*C.float)(unsafe.Pointer(&c[0])),
		n,
	)

	return RespErr.NdShape(c, er.t.Shape[0], t2.Shape[1])
}

func (t *Tensor) Matmul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Matmul(t2))
}
