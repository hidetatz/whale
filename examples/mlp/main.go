package main

import (
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/hidetatz/whale"
	tensor "github.com/hidetatz/whale/tensor2"
)

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

	for i := range 3000 {
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
	f, err := os.Create("cpu.prof")
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			panic(err)
		}
	}()
	if err := pprof.StartCPUProfile(f); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()
	xt, yt := whale.RandSin(100, 1)
	x := whale.NewVar(xt)
	y := whale.NewVar(yt)

	layer := [][]int{{1, 10}, {10, 1}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewMSE(), whale.NewSGD(0.2))
	Train(mlp, x, y)

	p := whale.NewPlot()
	if err := p.Scatter(xt.Flatten(), yt.Flatten(), "blue"); err != nil {
		panic(err)
	}

	t, err := tensor.Arange(0, 1, 0.01, 100, 1)
	if err != nil {
		panic(err)
	}
	tv := whale.NewVar(t)
	pred := Predict(mlp, tv)
	if err = p.Line(tv.GetData().Flatten(), pred.GetData().Flatten(), "red"); err != nil {
		panic(err)
	}

	if err := p.Exec(); err != nil {
		panic(err)
	}
}
