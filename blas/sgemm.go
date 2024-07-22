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

type sgemmParam struct {
	transA int
	transB int
	m      int
	n      int
	k      int
	alpha  float32
	a      []float32
	lda    int
	b      []float32
	ldb    int
	beta   float32
	c      []float32
	ldc    int
}

// Sgemm calculates a single precision general matrix multiplication.
// Because this blas package is a subproject of whale, this function assumes the row order is always row major as whale does so.
func Sgemm(transA, transB int, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	param := &sgemmParam{transA: transA, transB: transB, m: m, n: n, k: k, alpha: alpha, a: a, lda: lda, b: b, ldb: ldb, beta: beta, c: c, ldc: ldc}
	sgemmmain(param)
}

func dosgemm(param *sgemmParam, fn func(param *sgemmParam)) error {
	// todo: validate parameters
	fn(param)
	return nil
}

func sgemmmain(param *sgemmParam) {
	istrans := func(trans int) bool { return trans == Trans || trans == ConjTrans }

	transA := istrans(param.transA)
	transB := istrans(param.transB)

	switch {
	case !transA && !transB:
		// notrans x notrans.
		// C = α * A * B + β * C

		ab := float32(0.0)
		ai, bi, ci := 0, 0, 0
		for j := 0; j < param.n; j++ {
			for i := 0; i < param.m; i++ {
				ab = float32(0.0)
				for k := 0; k < param.k; k++ {
					ab = ab + param.a[ai]*param.b[bi]
					ai++
					bi += param.ldb
				}

				param.c[ci] = param.alpha*ab + param.beta*param.c[ci]
				ai = ai - param.k
				bi = bi - param.ldb*param.k + 1
				ci++
			}
			ai = ai + param.lda
			bi = bi - param.m
			ci = ci - param.m + param.ldc
		}

	case transA && !transB:
		// trans x notrans.
		// C = α * Aᵀ * B + β * C
		panic("unimplemented")

	case !transA && transB:
		// notrans x trans.
		// C = α * A * Bᵀ + β * C
		panic("unimplemented")

	default:
		// trans x trans.
		// C = α * Aᵀ * Bᵀ + β * C
		panic("unimplemented")
	}
}

func sgemmOpenBLAS_cgo(param *sgemmParam) {
	m := C.blasint(param.m)
	n := C.blasint(param.n)
	k := C.blasint(param.k)

	var transA uint32 = C.CblasNoTrans
	if param.transA == Trans {
		transA = C.CblasTrans
	}

	var transB uint32 = C.CblasNoTrans
	if param.transB == Trans {
		transB = C.CblasTrans
	}

	alpha := C.float(param.alpha)
	beta := C.float(param.beta)

	lda := C.blasint(param.lda)
	ldb := C.blasint(param.ldb)
	ldc := C.blasint(param.ldc)

	a := (*C.float)(unsafe.Pointer(&param.a[0]))
	b := (*C.float)(unsafe.Pointer(&param.b[0]))
	c := (*C.float)(unsafe.Pointer(&param.c[0]))

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
