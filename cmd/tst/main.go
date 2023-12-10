package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	// f := func(x *whale.Variable) *whale.Variable {
	// 	y := whale.Square_(x)
	// 	return y
	// }
	t1, err := tensor.Nd([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		panic(err)
	}

	t2, err := tensor.Nd([]float64{10, 20, 30}, 1, 3)
	if err != nil {
		panic(err)
	}

	x1 := whale.NewVar(t1)
	x2 := whale.NewVar(t2)

	y := whale.Add_(x1, x2)

	fmt.Println(y)

	// fmt.Println(x.GetData())
	// fmt.Println(x.GetGrad())
	// y := f(x)
	// fmt.Println(x.GetData())
	// fmt.Println(x.GetGrad())

	// y.Backward()
	// fmt.Println(x.GetData())
	// fmt.Println(x.GetGrad())

	// fmt.Println(y.GetData())
	// fmt.Println(y.GetGrad())
}
