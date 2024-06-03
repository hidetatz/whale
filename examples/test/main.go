package main

import (
	"fmt"
	"os"
	"runtime/pprof"
	"time"

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
