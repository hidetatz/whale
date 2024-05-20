package main

import (
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

	t1 := tensor.Arange(0, 100*784, 1).Reshape(100, 784).Copy()
	t2 := tensor.Arange(0, 784*1000, 1).Reshape(784, 1000).Copy()

	for range 600 {
		_ = t1.Matmul(t2)
	}
}
