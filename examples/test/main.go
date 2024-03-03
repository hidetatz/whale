package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor2"
)

func main() {
	t := tensor2.Must(tensor2.ArangeVec(1, 25, 1).Reshape(2, 3, 4))
	t2, err := t.Index(tensor2.At(0))
	if err != nil {
		panic(err)
	}
	fmt.Println(t2)
}
