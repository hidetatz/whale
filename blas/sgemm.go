package blas

import (
	/*
	   #cgo LDFLAGS: -L/opt/OpenBLAS/lib/ -lopenblas
	   #cgo CFLAGS: -I /opt/OpenBLAS/include/
	   #include <cblas.h>
	*/
	"C"
	"unsafe"
)

const (
	NoTrans   = 0
	Trans     = 1
	ConjTrans = 2

	ColMajor = 0
	RowMajor = 1
)

func Sgemm(order, transA, transB int, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	if order == RowMajor {
		// swap
		transA, transB = transB, transA
		m, n = n, m
		a, b = b, a
		lda, ldb = ldb, lda
	}

	// todo: validate params

	// memo:
	// L3 cache:   20MiB
	// L2 cache:   256KiB
	// L1 I cache: 32KiB
	// L1 D cache: 32KiB

	// todo: optimize
	// l3BlockSize := 1024
	// l2BlockSize := 128
	// l1BlockSize := 32

	transposed := func(trans int) bool { return trans == Trans || trans == ConjTrans }

	switch {
	case !transposed(transA) && !transposed(transB):
		// notrans x notrans.
		// C = α * A * B + β * C

		var ai, bi, ci int

		for _j := 0; _j < n; _j++ {
			for _i := 0; _i < m; _i++ {
				ab := float32(0.0)
				for _k := 0; _k < k; _k++ {
					ab = ab + a[ai]*b[bi]
					ai += lda
					bi++
				}
				c[ci] = beta*c[ci] + alpha*ab
				ai = ai - lda*k + 1
				bi = bi - k
				ci++
			}
			ai = ai - m
			bi = bi + ldb
			ci = ci - m + ldc
		}

		return nil

	case transposed(transA) && !transposed(transB):
		// trans x notrans.
		// C = α * A^T * B + β * C
		panic("unimplemented")

	case !transposed(transA) && transposed(transB):
		// notrans x trans.
		// C = α * A * B^T + β * C
		panic("unimplemented")

	default:
		// trans x trans.
		// C = α * A^T * B^T + β * C
		panic("unimplemented")
	}
}

func SgemmOpenBLAS_cgo(_order, _transA, _transB int, _m, _n, _k int, _alpha float32, _a []float32, _lda int, _b []float32, _ldb int, _beta float32, _c []float32, _ldc int) error {
	var order uint32 = C.CblasColMajor
	if _order == RowMajor {
		order = C.CblasRowMajor
	}
	var transA uint32 = C.CblasNoTrans
	if _transA == Trans || _transA == ConjTrans {
		transA = C.CblasTrans
	}

	var transB uint32 = C.CblasNoTrans
	if _transB == Trans || _transB == ConjTrans {
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
		order,
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
