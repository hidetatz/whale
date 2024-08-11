package blas

import (
	"math"
	"math/rand"
	"testing"
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
			c := randmatrix(t, size*size)
			c2 := make([]float32, len(c))
			copy(c2, c)

			openblasparam := &SgemmParam{
				TransA: transA,
				TransB: transB,
				M:      size,
				N:      size,
				K:      size,
				Alpha:  1,
				A:      a,
				LDA:    size,
				B:      b,
				LDB:    size,
				Beta:   1,
				C:      c,
				LDC:    size,
			}

			DoSgemm(openblasparam, SgemmOpenBLAS_cgo)

			param := &SgemmParam{
				TransA: transA,
				TransB: transB,
				M:      size,
				N:      size,
				K:      size,
				Alpha:  1,
				A:      a,
				LDA:    size,
				B:      b,
				LDB:    size,
				Beta:   1,
				C:      c2,
				LDC:    size,
			}

			DoSgemm(param, Sgemmmain)

			asserteq(t, openblasparam.C, param.C)
		}
	}
}
