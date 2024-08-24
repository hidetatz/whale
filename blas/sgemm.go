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

		for j := 0; j < n; j++ {
			for i := 0; i < m; i++ {
				c[ci] = beta * c[ci]
				ci++
			}
			ci = ci - m + ldc
		}
		ci = ci - ldc*n

		/*
		 * L3 cache
		 */
		for j3 := 0; j3 < n; j3 += min(n-j3, l3BlockSize) {
			for i3 := 0; i3 < m; i3 += min(m-i3, l3BlockSize) {
				for k3 := 0; k3 < k; k3 += min(k-k3, l3BlockSize) {

					/*
					 * L2 cache
					 */
					for j2 := j3; j2 < min(j3+l3BlockSize, n); j2 += min(n-j2, l2BlockSize) {
						for i2 := i3; i2 < min(i3+l3BlockSize, m); i2 += min(m-i2, l2BlockSize) {
							for k2 := k3; k2 < min(k3+l3BlockSize, k); k2 += min(k-k2, l2BlockSize) {

								/*
								 * L1 cache
								 */
								for j1 := j2; j1 < min(j2+l2BlockSize, n); j1 += min(n-j1, l1BlockSize) {
									for i1 := i2; i1 < min(i2+l2BlockSize, m); i1 += min(m-i1, l1BlockSize) {
										for k1 := k2; k1 < min(k2+l2BlockSize, k); k1 += min(k-k1, l1BlockSize) {

											ai = ai + lda*k1 + i1
											bi = bi + ldb*j1 + k1
											ci = ci + ldc*j1 + i1

											_m1 := min(l1BlockSize, m-i1)
											_n1 := min(l1BlockSize, n-j1)
											_k1 := min(l1BlockSize, k-k1)

											for j := j1; j < j1+_n1; j++ {
												for i := i1; i < i1+_m1; i++ {
													ab := float32(0.0)

													for k := k1; k < k1+_k1; k++ {
														ab = ab + a[ai]*b[bi]
														ai += lda
														bi++
													}

													c[ci] = c[ci] + alpha*ab
													ai = ai - lda*_k1 + 1
													bi = bi - _k1
													ci++
												}

												ai = ai - _m1
												bi = bi + ldb
												ci = ci - _m1 + ldc
											}

											bi = bi - ldb*_n1
											ci = ci - ldc*_n1

											ai = ai - lda*k1 - i1
											bi = bi - ldb*j1 - k1
											ci = ci - ldc*j1 - i1
										}
									}
								}
							}
						}
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
