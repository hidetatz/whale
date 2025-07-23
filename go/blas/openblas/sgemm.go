package openblas

/*
 * Sgemm implementation by OpenBLAS.
 * Defined for unit test.
 * This is located here, separated from blas package because
 * Go compiler does now allow cgo and asm exists in the same package.
 */

import (
	/*
	   #cgo LDFLAGS: -L/opt/OpenBLAS/lib/ -lopenblas
	   #cgo CFLAGS: -I /opt/OpenBLAS/include/
	   #include <cblas.h>
	*/
	"C"
	"unsafe"

	"github.com/hidetatz/whale/blas"
)

func Sgemm(_transA, _transB blas.Transpose, _m, _n, _k int, _alpha float32, _a []float32, _lda int, _b []float32, _ldb int, _beta float32, _c []float32, _ldc int) error {
	var transA uint32 = C.CblasNoTrans
	if _transA == blas.Trans || _transA == blas.ConjTrans {
		transA = C.CblasTrans
	}

	var transB uint32 = C.CblasNoTrans
	if _transB == blas.Trans || _transB == blas.ConjTrans {
		transB = C.CblasTrans
	}

	m := C.blasint(_m)
	n := C.blasint(_n)
	k := C.blasint(_k)

	alpha := C.float(_alpha)
	beta := C.float(_beta)

	lda := C.blasint(_lda)
	ldb := C.blasint(_ldb)
	ldc := C.blasint(_ldc)

	a := (*C.float)(unsafe.Pointer(&_a[0]))
	b := (*C.float)(unsafe.Pointer(&_b[0]))
	c := (*C.float)(unsafe.Pointer(&_c[0]))

	C.cblas_sgemm(
		C.CblasRowMajor,
		transA,
		transB,
		m,
		n,
		k,
		alpha,
		a,
		lda,
		b,
		ldb,
		beta,
		c,
		ldc,
	)

	return nil
}
