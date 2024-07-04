package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"
)

func main() {
	n := 1024

	a := make([][]float64, n)
	b := make([][]float64, n)
	c := make([][]float64, n)

	for i := range n {
		a[i] = make([]float64, n)
		b[i] = make([]float64, n)
		c[i] = make([]float64, n)
	}

	for i := range n {
		for j := range n {
			a[i][j] = rand.NormFloat64()
			b[i][j] = rand.NormFloat64()
		}
	}

	unroll := os.Getenv("UNROLL") == "1"
	fmt.Printf("unroll enabled: %v\n", unroll)

	start := time.Now()

	if unroll {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j += 8 {
				for k := 0; k < n; k++ {
					c[i][j] += a[i][k] * b[k][j]
					c[i][j+1] += a[i][k] * b[k][j+1]
					c[i][j+2] += a[i][k] * b[k][j+2]
					c[i][j+3] += a[i][k] * b[k][j+3]
					c[i][j+4] += a[i][k] * b[k][j+4]
					c[i][j+5] += a[i][k] * b[k][j+5]
					c[i][j+6] += a[i][k] * b[k][j+6]
					c[i][j+7] += a[i][k] * b[k][j+7]
				}
			}
		}
	} else {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				for k := 0; k < n; k++ {
					c[i][j] += a[i][k] * b[k][j]
					c[i][j] += a[i][k] * b[k][j]
				}
			}
		}
	}

	elapsed := time.Since(start)

	fmt.Printf("elapsed time = %v\n", elapsed)
}
