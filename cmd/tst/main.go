package main

import (
	"fmt"

	"github.com/hidetatz/whale"
)

func main() {
	// f := func(x []*whale.Variable) []*whale.Variable { return []*whale.Variable{whale.Add_(x[0], x[1])} }
	// x := []*whale.Variable{whale.NewVar(2), whale.NewVar(3)}
	// y := f(x)[0]
	f := func(x []*whale.Variable) []*whale.Variable {
		t1 := whale.Pow_(x[0], whale.NewVar(2))
		t2 := whale.Pow_(x[1], whale.NewVar(2))
		t3 := whale.Mul_(whale.NewVar(0.26), whale.Add_(t1, t2))
		t4 := whale.Mul_(whale.NewVar(0.48), x[0])
		t5 := whale.Mul_(t4, x[1])
		y := whale.Sub_(t3, t5)
		return []*whale.Variable{y}
	}
	x := []*whale.Variable{whale.NewVar(1), whale.NewVar(1)}
	y := f(x)[0]
	y.Backward()
	err := whale.VisualizeGraph(y)
	if err != nil {
		fmt.Println("error: ", err)
	}
}
