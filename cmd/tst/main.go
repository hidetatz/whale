package main

import (
	"fmt"
	"math"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

var device = &whale.CPU{}

func randdata() (*tensor.Tensor, *tensor.Tensor) {
	x := tensor.Rand(100, 1)
	y := device.Add(device.Sin(device.Mul(x, device.Mul(tensor.FromScalar(2), tensor.FromScalar(math.Pi)))), tensor.Rand(100, 1))
	return x, y
}

func Predict(m whale.Model, x *whale.Variable) *whale.Variable {
	pred, err := m.Train(x)
	if err != nil {
		panic(err)
	}
	return pred
}

func Train(m whale.Model, x, y *whale.Variable) {
	lossCalc := m.Loss()
	optim := m.Optimizer()

	for i := 0; i < 10000; i++ {
		pred, err := m.Train(x)
		if err != nil {
			panic(err)
		}

		loss, err := lossCalc.Calculate(pred, y)
		if err != nil {
			panic(err)
		}

		params := m.Params()
		for _, p := range params {
			p.ClearGrad()
		}

		loss.Backward()

		for _, p := range params {
			optim.Optimize(p)
		}

		if i%100 == 0 {
			fmt.Println(i)
		}

		if i%1000 == 0 {
			fmt.Println(loss)
		}
	}
}

func main() {
	xt, yt := randdata()
	x := whale.NewVar(xt)
	y := whale.NewVar(yt)

	layer := [][]int{{1, 10}, {10, 1}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewMSE(), whale.NewSGD(0.2))
	Train(mlp, x, y)

	p := whale.NewPlot()
	if err := p.Scatter(xt.Data, yt.Data, "blue"); err != nil {
		panic(err)
	}

	t, err := tensor.ArangeBy(0, 1, 0.01).Reshape(100, 1)
	if err != nil {
		panic(err)
	}
	tv := whale.NewVar(t)
	pred := Predict(mlp, tv)
	if err = p.Line(tv.GetData().Data, pred.GetData().Data, "red"); err != nil {
		panic(err)
	}

	if err := p.Exec(); err != nil {
		panic(err)
	}
}
