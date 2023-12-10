package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	t1, err := tensor.ArangeFrom(1, 4)
	if err != nil {
		panic(err)
	}

	t2, err := tensor.FromVector([]float64{10}, 1)
	if err != nil {
		panic(err)
	}

	x1 := whale.NewVar(t1)
	x2 := whale.NewVar(t2)
	y := whale.Add_(x1, x2)
	y.Backward()

	fmt.Println(y.GetData())
	fmt.Println(x1.GetGrad())
	fmt.Println(x2.GetGrad())
}
