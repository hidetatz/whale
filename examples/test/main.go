package main

import (
	"fmt"

	tensor "github.com/hidetatz/whale/tensor2"
)

func main() {
	t, err := tensor.Arange(0, 200, 1, 1, 5, 4, 2, 5)
	// t := tensor.ArangeVec(0, 5, 1)
	if err != nil {
		panic(err)
	}
	args := []*tensor.IndexArg{
		tensor.At(0),
		tensor.List(tensor.Vector([]float64{0, 1, 0})),
		tensor.At(2),
		tensor.At(1),
	}

	read := false

	if !read {
		err := t.IndexSub(args, tensor.Scalar(3))
		if err != nil {
			panic(err)
		}

		fmt.Println(t)
	} else {
		t2, err := t.Index(args...)

		if err != nil {
			panic(err)
		}

		fmt.Println(t2)
	}
}
