package main

import (
	"fmt"
	"time"

	"github.com/hidetatz/whale/blas"
	"github.com/hidetatz/whale/cpuid"
	"github.com/hidetatz/whale/flops"
)

func newmatrix(length int, val float32) []float32 {
	out := make([]float32, length)
	for i := range length {
		out[i] = val
	}
	return out
}

func main() {
	cpuinfo := cpuid.CPUID()
	flopsInfo := flops.Calc(cpuinfo)

	peakBase := flopsInfo.MFlopsFloatBase
	peakTurbo := flopsInfo.MFlopsFloatTurbo

	fmt.Printf("Max  Peak MFlops per core: %v MFlops\n", peakBase)
	fmt.Printf("Base Peak MFlops per core: %v MFlops\n", peakTurbo)
	fmt.Printf("size	elapsed time[s]	MFlops	base ratio[%%]	max ratio[%%]\n")
	for size := 16; size <= 2048; size *= 2 {
		// param := &blas.SgemmParam{
		order := blas.RowMajor
		transA := blas.NoTrans
		transB := blas.NoTrans
		m := size
		n := size
		k := size
		alpha := float32(1.0)
		a := newmatrix(size*size, 1)
		lda := size
		b := newmatrix(size*size, 1)
		ldb := size
		beta := float32(1.0)
		c := newmatrix(size*size, 0)
		ldc := size
		// }

		start := time.Now().UnixNano()
		err := blas.Sgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		finished := time.Now().UnixNano()
		elapsed := finished - start
		if err != nil {
			panic(err)
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
		theoreticalFlops := m * n * (2*k + 3)

		mflops := float64(theoreticalFlops) / (float64(elapsed) * 1e-9) / 1000.0 / 1000.0 * 1
		fmt.Printf("%v	%f	%f	%f	%f\n", size, float64(elapsed)*1e-9, mflops, mflops/peakBase*100, mflops/peakTurbo*100)
	}
}
