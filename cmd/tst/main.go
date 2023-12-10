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

	t1, err = t1.Reshape(3, 2, 4)
	if err != nil {
		panic(err)
	}

	x := whale.NewVar(t1)
	y := whale.SumAxes_(x, 1, 2, 0)

	fmt.Println(y.GetData())
}
