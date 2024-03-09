package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor2"
)

func main() {
	tensor, err := tensor2.ArangeVec(0, 48, 1).Reshape(2, 3, 4, 2)
	if err != nil {
		panic(err)
	}

	idx := []*tensor2.IndexArg{
		tensor2.List(tensor2.Must(tensor2.New([][]float64{{0, 0}, {0, 0}}))),
		tensor2.FromTo(0, 1),
		tensor2.At(1),
	}

	t2, err := tensor.Index(idx...)
	if err != nil {
		panic(err)
	}

	fmt.Println(t2, t2.Raw())
}
