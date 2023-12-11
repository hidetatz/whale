package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	t1, err := tensor.Nd([]float64{1, 2, 3, 1, 3, 2}, 2, 3)
	if err != nil {
		panic(err)
	}
	t2, err := tensor.Nd([]float64{1, 3, 5, 2, 4, 1, 1, 2, 3, 4, 5, 6}, 3, 4)
	if err != nil {
		panic(err)
	}

	x1 := whale.NewVar(t1)
	x2 := whale.NewVar(t2)
	y := whale.MatMul_(x1, x2)
	y.Backward()

	fmt.Println(x1.GetGrad())
	fmt.Println(x2.GetGrad())
}
