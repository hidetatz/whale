package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"time"

	"github.com/hidetatz/whale/tensor"
	"gonum.org/v1/gonum/mat"
)

func main() {
	f, err := os.Create("cpu.prof")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err := pprof.StartCPUProfile(f); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	t1 := tensor.Arange(0, 100*784, 1).Reshape(100, 784).Copy()
	t2 := tensor.Arange(0, 784*1000, 1).Reshape(784, 1000).Copy()

	start := time.Now()
	v1(t1, t2)
	elapsed := time.Since(start)
	fmt.Println("v1: ", elapsed)

	start = time.Now()
	v2(t1, t2)
	elapsed = time.Since(start)
	fmt.Println("v2: ", elapsed)

	start = time.Now()
	v3(t1, t2)
	elapsed = time.Since(start)
	fmt.Println("v3: ", elapsed)

	start = time.Now()
	gonum()
	elapsed = time.Since(start)
	fmt.Println("gonum: ", elapsed)
}

func v1(t1, t2 *tensor.Tensor) {
	for range 600 {
		_ = tensor.Matmul_v1(t1, t2)
	}
}

func v2(t1, t2 *tensor.Tensor) {
	for range 600 {
		_ = tensor.Matmul_v2(t1, t2)
	}
}

func v3(t1, t2 *tensor.Tensor) {
	for range 600 {
		_ = tensor.Matmul_v3(t1, t2)
	}
}

func gonum() {
	d1 := make([]float64, 100*784)
	for i := range 100 * 784 {
		d1[i] = float64(i)
	}

	d2 := make([]float64, 784*1000)
	for i := range 784 * 1000 {
		d2[i] = float64(i)
	}

	A := mat.NewDense(100, 784, d1)
	B := mat.NewDense(784, 1000, d2)

	C := mat.NewDense(100, 1000, nil)

	for range 600 {
		C.Mul(A, B)
	}
}
