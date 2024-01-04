package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor"
)

func main() {
	t, err := tensor.Arange(0, 16, 1, 2, 2, 4)
	if err != nil {
		panic(err)
	}

	// t2, err := t.Transpose(2, 1, 0)
	// if err != nil {
	// 	panic(err)
	// }

	fmt.Println(t)

	t3, err := t.SubTensor([]int{0, 1, 3})
	if err != nil {
		panic(err)
	}

	fmt.Println(t3)
}
