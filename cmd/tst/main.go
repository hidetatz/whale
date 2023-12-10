package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	t1, err := tensor.ArangeFrom(1, 25)
	if err != nil {
		panic(err)
	}

	t1, err = t1.Reshape(6, 4)
	if err != nil {
		panic(err)
	}

	x := whale.NewVar(t1)
	y := whale.SumTo_(x, 6, 1)

	fmt.Println(y.GetData())
}
