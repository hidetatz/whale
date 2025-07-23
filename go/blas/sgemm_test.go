package blas_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/hidetatz/whale/blas"
	"github.com/hidetatz/whale/blas/openblas"
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

func runSgemm(
	t *testing.T,
	fn func(transA, transB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) error,
	transA, transB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int,
) error {
	t.Helper()
	return fn(transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func TestSgemm(t *testing.T) {
	size := 1023
	transposes := []blas.Transpose{blas.NoTrans, blas.Trans}

	for _, transA := range transposes {
		for _, transB := range transposes {
			a := randmatrix(t, size*size)
			b := randmatrix(t, size*size)
			c := randmatrix(t, size*size)

			c2 := make([]float32, len(c))
			copy(c2, c)

			runSgemm(
				t,
				openblas.Sgemm,
				transA,
				transB,
				size,
				size,
				size,
				1,
				a,
				size,
				b,
				size,
				1,
				c,
				size,
			)

			runSgemm(
				t,
				blas.Sgemm,
				transA,
				transB,
				size,
				size,
				size,
				1,
				a,
				size,
				b,
				size,
				1,
				c2,
				size,
			)

			if len(c) != len(c2) {
				t.Fatalf("2 matrix has different length, %v, %v", len(c), len(c2))
			}

			// compare each value considering float rounding error.
			for i := range len(c) {
				rounderr := math.Abs(float64(c[i]) - float64(c2[i]))
				// Is there a better number?
				if float32(rounderr) > 0.001 {
					t.Fatalf("the value at %v has too big difference. ans: %v, calc: %v (transA: %v, transB: %v)", i, c[i], c2[i], transA, transB)
				}
			}
		}
	}
}
