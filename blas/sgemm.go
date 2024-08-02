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

	// memo:
	// L3 cache:   20MiB
	// L2 cache:   256KiB
	// L1 I cache: 32KiB
	// L1 D cache: 32KiB

	switch {
	case !transA && !transB:
		// notrans x notrans.
		// C = α * A * B + β * C

		ab := float32(0.0)
		ai, bi, ci := 0, 0, 0
		for j := 0; j < param.N; j++ {
			for i := 0; i < param.M; i++ {
				ab = float32(0.0)
				for k := 0; k < param.K; k++ {
					ab = ab + param.A[ai]*param.B[bi]
					ai++
					bi += param.LDB
				}

				param.C[ci] = param.Alpha*ab + param.Beta*param.C[ci]
				ai = ai - param.K
				bi = bi - param.LDB*param.K + 1
				ci++
			}
			ai = ai + param.LDA
			bi = bi - param.M
			ci = ci - param.M + param.LDC
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
