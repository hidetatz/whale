package blas

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/hidetatz/whale/cpuid"
	"github.com/hidetatz/whale/flops"
)

func newmatrix(t *testing.T, length int, val float32) []float32 {
	t.Helper()
	out := make([]float32, length)
	for i := range length {
		out[i] = val
	}
	return out
}

func randmatrix(t *testing.T, length int) []float32 {
	t.Helper()
	out := make([]float32, length)
	for i := range length {
		out[i] = float32(rand.NormFloat64())
	}
	return out
}

func asserteq(t *testing.T, ans, got []float32) {
	t.Helper()
	if len(ans) != len(got) {
		t.Fatalf("2 matrix has different length, %v, %v", len(ans), len(got))
	}

	// compare each value considering float rounding error.
	for i := range len(ans) {
		rounderr := math.Abs(float64(ans[i]) - float64(got[i]))
		// Is there a better number?
		if float32(rounderr) > 0.001 {
			t.Fatalf("the value at %v has too big difference. ans: %v, calc: %v", i, ans[i], got[i])
		}
	}
}

func TestSgemm(t *testing.T) {
	size := 1023
	transpose := []int{NoTrans} // currently notrans x notrans is only implemented

	for _, transA := range transpose {
		for _, transB := range transpose {
			a := randmatrix(t, size*size)
			b := randmatrix(t, size*size)

			openblasparam := &sgemmParam{
				transA: transA,
				transB: transB,
				m:      size,
				n:      size,
				k:      size,
				alpha:  1,
				a:      a,
				lda:    size,
				b:      b,
				ldb:    size,
				beta:   1,
				c:      newmatrix(t, size*size, 0),
				ldc:    size,
			}

			dosgemm(openblasparam, sgemmOpenBLAS_cgo)

			param := &sgemmParam{
				transA: transA,
				transB: transB,
				m:      size,
				n:      size,
				k:      size,
				alpha:  1,
				a:      a,
				lda:    size,
				b:      b,
				ldb:    size,
				beta:   1,
				c:      newmatrix(t, size*size, 0),
				ldc:    size,
			}

			dosgemm(param, sgemmmain)

			asserteq(t, openblasparam.c, param.c)
		}
	}
}

func TestSgemm_perf(t *testing.T) {
	cpuinfo := cpuid.CPUID()
	flopsInfo := flops.Calc(cpuinfo)

	peakBase := flopsInfo.MFlopsDoubleTurbo / float64(cpuinfo.LogicalCores)
	peakTurbo := flopsInfo.MFlopsDoubleBase / float64(cpuinfo.LogicalCores)

	t.Logf("Max  Peak MFlops per core: %v MFlops\n", peakBase)
	t.Logf("Base Peak MFlops per core: %v MFlops\n", peakTurbo)
	for size := 16; size <= 2048; size *= 2 {
		param := &sgemmParam{
			transA: NoTrans,
			transB: NoTrans,
			m:      size,
			n:      size,
			k:      size,
			alpha:  1,
			a:      newmatrix(t, size*size, 1),
			lda:    size,
			b:      newmatrix(t, size*size, 1),
			ldb:    size,
			beta:   1,
			c:      newmatrix(t, size*size, 0),
			ldc:    size,
		}

		start := time.Now()
		err := dosgemm(param, sgemmmain)
		elapsed := time.Since(start)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		/*
		 * Estimated float operation number per a sgemm.
		 * Typical sgemm (C = alpha * A * B + beta * C) looks like this:
		 *
		 * for( j=0; j<N; j++){
		 *     for( i=0; i<M; i++){
		 *         ab = 0;
		 *         for( k=0; k<K; k++){
		 *             ab = ab + A[i][k]*B[k][j];
		 *         }
		 *         C[i][j] = alpha*ab + beta*C[i][j];
		 *     }
		 * }
		 *
		 * The most inside loop does 2 operations (1 MUL and 1 ADD). This happens for M*N*K times.
		 * And the write to C does 3 operations (2 MUL and 1 ADD). This happens for M*N times.
		 * Total operation count would be 2*M*N*K + 3*M*N = M*N*(2*K+3).
		 */
		theoriticalFlops := param.m * param.n * (2*param.k + 3)

		mflops := theoriticalFlops / int(elapsed) / 1000 / 1000
		t.Logf("%v	%v	%v	%v	%v\n", size, elapsed, mflops, mflops/int(peakBase)*100, mflops/int(peakTurbo)*100)
	}
}
