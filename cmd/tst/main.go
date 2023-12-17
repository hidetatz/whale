package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func p(v *whale.Variable) {
	fmt.Println(v.GetData())
	fmt.Println(v.GetGrad())
}

func main() {
	t1, err := tensor.Nd([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		panic(err)
	}

	t2, err := tensor.Nd([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4)
	if err != nil {
		panic(err)
	}
	x0 := whale.NewVar(t1)
	x1 := whale.NewVar(t2)
	y := whale.MatMul_(x0, x1)
	y.Backward()

	p(y)
	p(x0)
	p(x1)
}

func main2() {
	wt, err := tensor.Nd([]float64{1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12}, 4, 3)
	if err != nil {
		panic(err)
	}
	xt, err := tensor.Nd([]float64{1, 4, 2, 5, 3, 6}, 3, 2)
	if err != nil {
		panic(err)
	}

	y, err := tensor.Nd([]float64{1, 1, 1, 1, 1, 1, 1, 1}, 2, 4)
	if err != nil {
		panic(err)
	}

	cpu := &whale.CPU{}
	fmt.Println(cpu.Dot(y, wt))
	fmt.Println(cpu.Dot(xt, y))
}
