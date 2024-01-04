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

	t2, err := t.Transpose(2, 1, 0)
	if err != nil {
		panic(err)
	}

	fmt.Println(t2)
}
