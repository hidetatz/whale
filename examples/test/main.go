package main

import (
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/hidetatz/whale/tensor"
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

	t1 := tensor.New(
		[][]float64{
			{8, 4, 2},
			{1, 3, -6},
			{-7, 0, 5},
		},
	)

	t2 := tensor.New(
		[][]float64{
			{5, 2},
			{3, 1},
			{4, -1},
		},
	)

	// t1 := tensor.Arange(0, 100*784, 1).Reshape(100, 784)
	// t2 := tensor.Arange(0, 100*784, 1).Reshape(784, 100)
	// t3 := tensor.Arange(0, 100*10, 1).Reshape(100, 10)

	//for _ = range 600 {
	fmt.Println(t1.Matmul(t2))
	//		r2 := r.Dot(t3)
	//		_, _ = r, r2
	//	}
}
