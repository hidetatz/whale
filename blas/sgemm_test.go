package blas

import (
	"math"
	"math/rand"
	"testing"
	"time"
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

func asserteq(t *testing.T, ans, got []float32, k int) {
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

			dosgemm(openblasparam, sgemmOpenBLAS)

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

			asserteq(t, openblasparam.c, param.c, param.k)
		}
	}
}

func checkspeed(t *testing.T, param *sgemmParam, fn func(param *sgemmParam)) time.Duration {
	t.Helper()

	start := time.Now()
	err := dosgemm(param, fn)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	return elapsed
}

func TestSgemm_perf(t *testing.T) {
	// todo:
}
