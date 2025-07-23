package main

import (
	"fmt"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	t := tensor.Arange(0, 16, 1).Reshape(4, 4)
	fmt.Println(t.Raw())
	fmt.Println(t)
	t2 := t.Transpose()
	fmt.Println(t2.Raw())
	fmt.Println(t2)
}
