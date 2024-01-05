package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor"
)

func main() {
	t, err := tensor.Arange(1, 25, 1, 2, 3, 4)
	if err != nil {
		panic(err)
	}

	// t2, err := t.Transpose(2, 1, 0)
	// if err != nil {
	// 	panic(err)
	// }

	fmt.Println(t)

	t3, err := t.Tile(2, 2, 2, 2)
	if err != nil {
		panic(err)
	}

	fmt.Println(t3)
}
