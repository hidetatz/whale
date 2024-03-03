package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor2"
)

func main() {
	// t := tensor2.Must(tensor2.ArangeVec(1, 25, 1).Reshape(2, 3, 4))
	// t2 := tensor2.Must(t.Index(tensor2.At(1)))

	// fmt.Println(t2)
	// err := t2.IndexAdd(3, tensor2.At(0), tensor2.At(2))
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(t2)

	tensor, err := tensor2.New([][]float64{{1, 2}, {3, 4}})
	if err != nil {
		panic(err)
	}
	fmt.Println(tensor.Raw())
	err = tensor.IndexSub(2, tensor2.At(0))
	if err != nil {
		panic(err)
	}
	fmt.Println(tensor)
}
