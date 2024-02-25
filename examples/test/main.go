package main

import (
	"fmt"

	"github.com/hidetatz/whale/tensor2"
)

func main() {
	t := tensor2.Must(tensor2.ArangeVec(1, 25, 1).Reshape(2, 3, 4))
	s, err := t.Sum(true)
	if err != nil {
		panic(err)
	}
	fmt.Println(s)
}
