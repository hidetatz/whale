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
)

type SgemmParam struct {
	TransA int
	TransB int
	M      int
	N      int
	K      int
	Alpha  float32
	A      []float32
	LDA    int
	B      []float32
	LDB    int
	Beta   float32
	C      []float32
	LDC    int
}

// Sgemm calculates a single precision general matrix multiplication.
// Because this blas package is a subproject of whale, this function assumes the row order is always row major as whale does so.
func Sgemm(transA, transB int, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error {
	param := &SgemmParam{TransA: transA, TransB: transB, M: m, N: n, K: k, Alpha: alpha, A: a, LDA: lda, B: b, LDB: ldb, Beta: beta, C: c, LDC: ldc}
	return DoSgemm(param, Sgemmmain)
}

func DoSgemm(param *SgemmParam, fn func(param *SgemmParam)) error {
	// todo: validate parameters
	fn(param)
	return nil
}

var Sgemmmain = func(param *SgemmParam) {
	istrans := func(trans int) bool { return trans == Trans || trans == ConjTrans }

	transA := istrans(param.TransA)
	transB := istrans(param.TransB)

	M := param.M
	N := param.N
	K := param.K
	Alpha := param.Alpha
	LDA := param.LDA
	LDB := param.LDB
	Beta := param.Beta
	LDC := param.LDC

	// memo:
	// L3 cache:   20MiB
	// L2 cache:   256KiB
	// L1 I cache: 32KiB
	// L1 D cache: 32KiB

	// todo: optimize
	l3BlockSize := 1024
	l2BlockSize := 128
	l1BlockSize := 32

	switch {
	case !transA && !transB:
		// notrans x notrans.
		// C = α * A * B + β * C

		var ai, bi, ci int

		// scale C + beta outside the loop
		for j := 0; j < N; j++ {
			for i := 0; i < M; i++ {
				param.C[ci] = Beta * param.C[ci]
				ci++
			}
			ci = ci - M + LDC
		}
		ci = ci - LDC*N

		/*
		 * j-loop strip mining
		 */

		for j3 := 0; j3 < N; j3 += min(N-j3, l3BlockSize) {
			for j2 := j3; j2 < min(j3+l3BlockSize, N); j2 += min(N-j2, l2BlockSize) {
				for j1 := j2; j1 < min(j2+l2BlockSize, N); j1 += min(N-j1, l1BlockSize) {
					for j := j1; j < min(j1+l1BlockSize, N); j++ {

						/*
						 * i-loop strip mining
						 */

						for i3 := 0; i3 < M; i3 += min(M-i3, l3BlockSize) {
							for i2 := i3; i2 < min(i3+l3BlockSize, M); i2 += min(M-i2, l2BlockSize) {
								for i1 := i2; i1 < min(i2+l2BlockSize, M); i1 += min(M-i1, l1BlockSize) {
									for i := i1; i < min(i1+l1BlockSize, M); i++ {

										/*
										 * k-loop strip mining
										 */

										ab := float32(0.0)

										for k3 := 0; k3 < K; k3 += min(K-k3, l3BlockSize) {
											for k2 := k3; k2 < min(k3+l3BlockSize, K); k2 += min(K-k2, l2BlockSize) {
												for k1 := k2; k1 < min(k2+l2BlockSize, K); k1 += min(K-k1, l1BlockSize) {
													for k := k1; k < min(k1+l1BlockSize, K); k++ {
														ab = ab + param.A[ai]*param.B[bi]
														ai++
														bi += LDB
													}
												}
											}
										}
										param.C[ci] = Alpha*ab + param.C[ci]
										ai = ai - param.K
										bi = bi - param.LDB*param.K + 1
										ci++
									}
								}
							}
						}
						ai = ai + LDA
						bi = bi - M
						ci = ci - M + LDC
					}
				}
			}
		}

	case transA && !transB:
		// trans x notrans.
		// C = α * A^T * B + β * C
		panic("unimplemented")

	case !transA && transB:
		// notrans x trans.
		// C = α * A * B^T + β * C
		panic("unimplemented")

	default:
		// trans x trans.
		// C = α * A^T * B^T + β * C
		panic("unimplemented")
	}
}

var SgemmOpenBLAS_cgo = func(param *SgemmParam) {
	m := C.blasint(param.M)
	n := C.blasint(param.N)
	k := C.blasint(param.K)

	var transA uint32 = C.CblasNoTrans
	if param.TransA == Trans {
		transA = C.CblasTrans
	}

	var transB uint32 = C.CblasNoTrans
	if param.TransB == Trans {
		transB = C.CblasTrans
	}

	alpha := C.float(param.Alpha)
	beta := C.float(param.Beta)

	lda := C.blasint(param.LDA)
	ldb := C.blasint(param.LDB)
	ldc := C.blasint(param.LDC)

	a := (*C.float)(unsafe.Pointer(&param.A[0]))
	b := (*C.float)(unsafe.Pointer(&param.B[0]))
	c := (*C.float)(unsafe.Pointer(&param.C[0]))

	C.cblas_sgemm(
		C.CblasRowMajor, // assume row major always because whale uses row major
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
}
