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

const (
	I     = 1
	H     = 10
	O     = 1
	LR    = 0.2
	iters = 20000
)

var xt, yt = randdata()
var x = whale.NewVar(xt)
var y = whale.NewVar(yt)
var w1 = whale.NewVar(device.Mul(tensor.Rand(I, H), tensor.FromScalar(0.01)))
var b1 = whale.NewVar(tensor.Zeros(H))
var w2 = whale.NewVar(device.Mul(tensor.Rand(H, O), tensor.FromScalar(0.01)))
var b2 = whale.NewVar(tensor.Zeros(O))

func main() {
	for i := 0; i < iters; i++ {
		yPred := predict(x)
		loss, err := whale.MeanSquaredError(y, yPred)
		if err != nil {
			panic(err)
		}

		w1.ClearGrad()
		b1.ClearGrad()
		w2.ClearGrad()
		b2.ClearGrad()
		loss.Backward()

		w1.Sub(LR)
		b1.Sub(LR)
		w2.Sub(LR)
		b2.Sub(LR)

		if i%100 == 0 {
			fmt.Println(i)
		}

		if i%1000 == 0 {
			fmt.Println(loss)
		}
	}

	whale.Scatter(xt.Data, yt.Data)

	t, err := tensor.ArangeBy(0, 1, 0.01).Reshape(100, 1)
	if err != nil {
		panic(err)
	}
	tv := whale.NewVar(t)
	pred := predict(tv)
	if err = whale.Scatter(tv.GetData().Data, pred.GetData().Data); err != nil {
		panic(err)
	}
}

func predict(x *whale.Variable) *whale.Variable {
	y, err := whale.Linear(x, w1, b1)
	if err != nil {
		panic(err)
	}
	y, err = whale.Sigmoid(y)
	if err != nil {
		panic(err)
	}
	y, err = whale.Linear(y, w2, b2)
	if err != nil {
		panic(err)
	}

	return y
}
