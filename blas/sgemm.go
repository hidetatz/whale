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
	l3BlockSize := 1024
	l2BlockSize := 128
	l1BlockSize := 32

	transposed := func(trans int) bool { return trans == Trans || trans == ConjTrans }

	switch {
	case !transposed(transA) && !transposed(transB):
		// notrans x notrans.
		// C = α * A * B + β * C

		var ai, bi, ci int

		/*
		 * j-loop strip mining
		 */

		for _j3 := 0; _j3 < n; _j3 += min(n-_j3, l3BlockSize) {
			for _j2 := _j3; _j2 < min(_j3+l3BlockSize, n); _j2 += min(n-_j2, l2BlockSize) {
				for _j1 := _j2; _j1 < min(_j2+l2BlockSize, n); _j1 += min(n-_j1, l1BlockSize) {
					for _j := _j1; _j < min(_j1+l1BlockSize, n); _j++ {

						/*
						 * i-loop strip mining
						 */

						for _i3 := 0; _i3 < m; _i3 += min(m-_i3, l3BlockSize) {
							for _i2 := _i3; _i2 < min(_i3+l3BlockSize, m); _i2 += min(m-_i2, l2BlockSize) {
								for _i1 := _i2; _i1 < min(_i2+l2BlockSize, m); _i1 += min(m-_i1, l1BlockSize) {
									for _i := _i1; _i < min(_i1+l1BlockSize, m); _i++ {

										ab := float32(0.0)

										/*
										 * k-loop strip mining
										 */

										for _k3 := 0; _k3 < k; _k3 += min(k-_k3, l3BlockSize) {
											for _k2 := _k3; _k2 < min(_k3+l3BlockSize, k); _k2 += min(k-_k2, l2BlockSize) {
												for _k1 := _k2; _k1 < min(_k2+l2BlockSize, k); _k1 += min(k-_k1, l1BlockSize) {
													for _k := _k1; _k < min(_k1+l1BlockSize, k); _k++ {

														ab = ab + a[ai]*b[bi]
														ai += lda
														bi++

													}
												}
											}
										}

										c[ci] = beta*c[ci] + alpha*ab
										ai = ai - lda*k + 1
										bi = bi - k
										ci++

									}
								}
							}
						}

						ai = ai - m
						bi = bi + ldb
						ci = ci - m + ldc

					}
				}
			}
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
