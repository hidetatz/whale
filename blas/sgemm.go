package blas

/*
 * Sgemm implementation is mostly copied from gonum (https://www.gonum.org/),
 * but some modifications are applied such as:
 * - dynamic blocking size optimization
 * - error handling
 * - do some whale-style simplification
 * - comments
 * The original gonum license is below:
 *
 * Copyright ©2014 The Gonum Authors. All rights reserved.
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 *
 * Assembly source files are usually just copied from gonum,
 * but some modifications are applied such as:
 * - build tag change
 * - file name
 * The license should be referred to the .s files.
 */

import (
	"fmt"
	"runtime"
	"sync"
)

type Transpose int

const (
	NoTrans   Transpose = 0
	Trans     Transpose = 1
	ConjTrans Transpose = 2
)

func (t Transpose) String() string {
	switch t {
	case 0:
		return "NoTrans"
	case 1:
		return "Trans"
	case 2:
		return "ConjTrans"
	default:
		panic("transpose.String(): unreachable")
	}
}

// implemented in .s
func SaxpyUnitary(alpha float32, x, y []float32)
func SaxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)
func DotUnitary(x, y []float32) (sum float32)

const (
	blockSize   = 512
	minParBlock = 4
)

// Sgemm is a blas sgemm implementation. See blas reference for each argument meaning.
// In this Sgemm, order is assumed to be row-major because this blas lib is made for whale
// which always considers tensors as row-major.
func Sgemm(
	transA, transB Transpose, m, n, k int,
	alpha float32, a []float32, lda int, b []float32, ldb int,
	beta float32, c []float32, ldc int) error {

	/*
	 * validation
	 */

	if transA != NoTrans && transA != Trans && transA != ConjTrans {
		return fmt.Errorf("invalid transpose %v for A", transA)
	}

	if transB != NoTrans && transB != Trans && transB != ConjTrans {
		return fmt.Errorf("invalid transpose %v for B", transB)
	}

	isATrans := transA == Trans || transA == ConjTrans
	isBTrans := transB == Trans || transB == ConjTrans

	if m < 0 {
		return fmt.Errorf("m < 0")
	}

	if n < 0 {
		return fmt.Errorf("n < 0")
	}

	if k < 0 {
		return fmt.Errorf("k < 0")
	}

	if isATrans {
		if lda < max(1, m) {
			return fmt.Errorf("invalid lda")
		}
	} else {
		if lda < max(1, k) {
			return fmt.Errorf("invalid lda")
		}
	}

	if isBTrans {
		if ldb < max(1, k) {
			return fmt.Errorf("invalid ldb")
		}
	} else {
		if ldb < max(1, n) {
			return fmt.Errorf("invalid ldb")
		}
	}

	if ldc < max(1, n) {
		return fmt.Errorf("invalid ldc")
	}

	// lucky, fast path

	if m == 0 || n == 0 {
		return nil
	}

	if alpha == 0 && beta == 1 {
		return nil
	}

	// c *= beta
	if beta != 1 {
		for i := 0; i < m; i++ {
			tmp := c[i*ldc : i*ldc+n] // tmp is a view, not a copy
			for j := range tmp {
				tmp[j] *= beta
			}
		}
	}

	/*
	 * Actual sgemm computation.
	 * The basic performance tuning ideas are:
	 * - blocking
	 * - parallelization
	 * - SIMD
	 * - loop unrolling
	 */

	// sgemmParallel computes a parallel matrix multiplication bys partitioning
	// a and b into sub-blocks, and updating c with the multiplication of the sub-block
	// In all cases,
	// A = [ 	A_11	A_12 ... 	A_1j
	//			A_21	A_22 ...	A_2j
	//				...
	//			A_i1	A_i2 ...	A_ij]
	//
	// and same for B. All of the submatrix sizes are blockSize×blockSize except
	// at the edges.
	//
	// In all cases, there is one dimension for each matrix along which
	// C must be updated sequentially.
	// Cij = \sum_k Aik Bkj,	(A * B)
	// Cij = \sum_k Aki Bkj,	(Aᵀ * B)
	// Cij = \sum_k Aik Bjk,	(A * Bᵀ)
	// Cij = \sum_k Aki Bjk,	(Aᵀ * Bᵀ)
	//
	// This code computes one {i, j} block sequentially along the k dimension,
	// and computes all of the {i, j} blocks concurrently. This
	// partitioning allows Cij to be updated in-place without race-conditions.
	// Instead of launching a goroutine for each possible concurrent computation,
	// a number of worker goroutines are created and channels are used to pass
	// available and completed cases.
	//
	// http://alexkr.com/docs/matrixmult.pdf is a good reference on matrix-matrix
	// multiplies, though this code does not copy matrices to attempt to eliminate
	// cache misses.

	maxKLen := k
	parBlocks := blocks(m, blockSize) * blocks(n, blockSize)
	if parBlocks < minParBlock {
		// matrix is small enough
		sgemm(isATrans, isBTrans, m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return nil
	}

	workerLimit := make(chan struct{}, runtime.GOMAXPROCS(0))

	var wg sync.WaitGroup
	wg.Add(parBlocks)
	defer wg.Wait()

	for i := 0; i < m; i += blockSize {
		for j := 0; j < n; j += blockSize {
			workerLimit <- struct{}{}
			go func(i, j int) {
				defer func() {
					wg.Done()
					<-workerLimit
				}()

				leni := blockSize
				if i+leni > m {
					leni = m - i
				}
				lenj := blockSize
				if j+lenj > n {
					lenj = n - j
				}

				cSub := sliceView(c, ldc, i, j, leni, lenj)

				// Compute A_ik B_kj for all k
				for k := 0; k < maxKLen; k += blockSize {
					lenk := blockSize
					if k+lenk > maxKLen {
						lenk = maxKLen - k
					}
					var aSub, bSub []float32
					if isATrans {
						aSub = sliceView(a, lda, k, i, lenk, leni)
					} else {
						aSub = sliceView(a, lda, i, k, leni, lenk)
					}
					if isBTrans {
						bSub = sliceView(b, ldb, j, k, lenj, lenk)
					} else {
						bSub = sliceView(b, ldb, k, j, lenk, lenj)
					}

					sgemm(isATrans, isBTrans, leni, lenj, lenk, aSub, lda, bSub, ldb, cSub, ldc, alpha)
				}
			}(i, j)
		}
	}

	return nil
}

// sgemm computes a straightforward matrix multiplication sequentially.
func sgemm(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	switch {
	case !aTrans && !bTrans:
		for i := 0; i < m; i++ {
			ctmp := c[i*ldc : i*ldc+n]
			for l, v := range a[i*lda : i*lda+k] {
				tmp := alpha * v
				if tmp != 0 {
					SaxpyUnitary(tmp, b[l*ldb:l*ldb+n], ctmp)
				}
			}
		}
	case aTrans && !bTrans:
		for l := 0; l < k; l++ {
			btmp := b[l*ldb : l*ldb+n]
			for i, v := range a[l*lda : l*lda+m] {
				tmp := alpha * v
				if tmp != 0 {
					ctmp := c[i*ldc : i*ldc+n]
					SaxpyUnitary(tmp, btmp, ctmp)
				}
			}
		}
	case !aTrans && bTrans:
		for i := 0; i < m; i++ {
			atmp := a[i*lda : i*lda+k]
			ctmp := c[i*ldc : i*ldc+n]
			for j := 0; j < n; j++ {
				ctmp[j] += alpha * DotUnitary(atmp, b[j*ldb:j*ldb+k])
			}
		}

	default: // aTrans && bTrans:
		for l := 0; l < k; l++ {
			for i, v := range a[l*lda : l*lda+m] {
				tmp := alpha * v
				if tmp != 0 {
					ctmp := c[i*ldc : i*ldc+n]
					SaxpyInc(tmp, b[l:], ctmp, uintptr(n), uintptr(ldb), 1, 0, 0)
				}
			}
		}
	}
}

func blocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

func sliceView(a []float32, lda, i, j, r, c int) []float32 {
	return a[i*lda+j : (i+r-1)*lda+j+c]
}
