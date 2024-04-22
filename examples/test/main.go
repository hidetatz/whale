package main

import (
	"fmt"

	tensor "github.com/hidetatz/whale/tensor2"
)

func main() {
	t := tensor.Rand(2, 2, 2)
	// t2, err := t.Index(
	// 	tensor.FromTo(0, 2),
	// 	tensor.At(3),
	// 	tensor.FromTo(1, 6),
	// 	tensor.List(tensor.Vector([]float64{1, 5, 2})),
	// 	tensor.List(tensor.Vector([]float64{0, 0, 0})),
	// )

	// if err != nil {
	// 	panic(err)
	// }

	fmt.Println(t.String())
	fmt.Println(t.OnelineString())
}
