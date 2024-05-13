package main

import (
	"fmt"

	tensor "github.com/hidetatz/whale/tensor2"
)

func main() {
	t := tensor.Arange(0, 200, 1, 1, 5, 4, 2, 5)
	t2, err := t.Index(
		tensor.At(0),
		tensor.At(1),
	)
	if err != nil {
		panic(err)
	}

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
		t3, err := t2.Index(args...)

		if err != nil {
			panic(err)
		}

		fmt.Println(t3)
	}
}
