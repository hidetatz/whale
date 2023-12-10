package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	f := func(x *whale.Variable) *whale.Variable {
		y := whale.Square_(x)
		return y
	}

	x := whale.NewVar(tensor.FromScalar(2))
	fmt.Println(x.GetData())
	fmt.Println(x.GetGrad())
	y := f(x)
	fmt.Println(x.GetData())
	fmt.Println(x.GetGrad())

	y.Backward()
	fmt.Println(x.GetData())
	fmt.Println(x.GetGrad())

	fmt.Println(y.GetData())
	fmt.Println(y.GetGrad())
}
