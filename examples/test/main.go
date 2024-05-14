package main

import (
	"fmt"

	tensor "github.com/hidetatz/whale/tensor2"
)

func main() {
	t := tensor.Arange(0, 200, 1, 1, 5, 4, 2, 5)
	t2 := t.Index(tensor.At(0), tensor.At(1))

	fmt.Println(t2)

	args := []*tensor.IndexArg{
		tensor.At(0),
		tensor.List(tensor.Vector([]float64{0, 1, 0})),
	}

	read := false

	if !read {
		err := t2.IndexSub(args, tensor.Scalar(3))
		if err != nil {
			panic(err)
		}

		fmt.Println(t2)
	} else {
		t3 := t2.Index(args...)
		fmt.Println(t3)
	}
}
