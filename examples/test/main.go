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

	fmt.Println(time.Now())
	ten()
	fmt.Println(time.Now())
	gonum()
	fmt.Println(time.Now())
}

func ten() {
	t1 := tensor.Arange(0, 100*784, 1).Reshape(100, 784).Copy()
	t2 := tensor.Arange(0, 784*1000, 1).Reshape(784, 1000).Copy()

	for range 600 {
		_ = t1.Matmul(t2)
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
