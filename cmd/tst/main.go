package main

import (
	"fmt"

	"github.com/hidetatz/whale"
)

func main() {
	// f := func(x []*whale.Variable) []*whale.Variable { return []*whale.Variable{whale.Add_(x[0], x[1])} }
	// x := []*whale.Variable{whale.NewVar(2), whale.NewVar(3)}
	// y := f(x)[0]
	// f := func(x []*whale.Variable) []*whale.Variable {
	// 	t1 := whale.Pow_(x[0], whale.NewVar(2))
	// 	t2 := whale.Pow_(x[1], whale.NewVar(2))
	// 	t3 := whale.Mul_(whale.NewVar(0.26), whale.Add_(t1, t2))
	// 	t4 := whale.Mul_(whale.NewVar(0.48), x[0])
	// 	t5 := whale.Mul_(t4, x[1])
	// 	y := whale.Sub_(t3, t5)
	// 	return []*whale.Variable{y}
	// }
	// x := []*whale.Variable{whale.NewVar(1), whale.NewVar(1)}
	// y := f(x)[0]
	// y.Backward()
	// err := whale.VisualizeGraph(y)
	// if err != nil {
	// 	fmt.Println("error: ", err)
	// }

	// f := func(x []*whale.Variable) []*whale.Variable {
	// 	t1 := whale.Pow_(x[0], whale.NewVar(4))
	// 	t2 := whale.Pow_(x[0], whale.NewVar(2))
	// 	t3 := whale.Mul_(whale.NewVar(2), t2)
	// 	y := whale.Sub_(t1, t3)
	// 	return []*whale.Variable{y}
	// }
	// x := []*whale.Variable{whale.NewVar(2)}

	// for i := 0; i < 10; i++ {
	// 	fmt.Println(i, x[0])
	// 	y := f(x)[0]
	// 	x[0].ClearGrad()
	// 	y.Backward()

	// 	// fmt.Println(x[0].GetGrad())

	// 	gx := x[0].GetGrad()
	// 	x[0].ClearGrad()
	// 	gx.Backward()
	// 	gx2 := x[0].GetGrad()

	// 	x[0].SetData(x[0].GetData() - (gx.GetData() / gx2.GetData()))
	// }

	// err := whale.VisualizeGraph(y)
	// if err != nil {
	// 	fmt.Println("error: ", err)
	// }

	// f := func(x []*whale.Variable) []*whale.Variable {
	// 	y := whale.Sin_(x[0])
	// 	return []*whale.Variable{y}
	// }
	// x := []*whale.Variable{whale.NewVar(1)}

	// y := f(x)[0]
	// y.Backward()
	// for i := 0; i < 3; i++ {
	// 	gx := x[0].GetGrad()
	// 	x[0].ClearGrad()
	// 	gx.Backward()
	// 	fmt.Println(x[0].GetGrad())
	// }

	f := func(x []*whale.Variable) []*whale.Variable {
		y := whale.Tanh_(x[0])
		return []*whale.Variable{y}
	}
	x := []*whale.Variable{whale.NewVar(1)}

	y := f(x)[0]
	y.Backward()

	iter := 1
	for i := 0; i < iter; i++ {
		gx := x[0].GetGrad()
		x[0].ClearGrad()
		gx.Backward()
	}

	gx := x[0].GetGrad()
	err := whale.VisualizeGraph(gx)
	if err != nil {
		fmt.Println("error: ", err)
	}
}
