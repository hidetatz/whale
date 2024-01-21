package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main2() {
	x := whale.NewVar(tensor.MustNd([]float64{0.2, -0.4, 0.3, 0.5, 1.3, -3.2, 2.1, 0.3}, 4, 2))
	t := whale.NewVar(tensor.Vector([]float64{2, 0, 1, 0}))

	layer := [][]int{{2, 10}, {10, 3}}
	mlp := whale.NewMLP(layer, true, whale.NewSoftMax(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(0.2))

	pred, err := mlp.Train(x)
	if err != nil {
		panic(err)
	}

	fmt.Println(pred)

	p, err := whale.NewSoftMax().Activate(pred)
	fmt.Println(p)

	loss, err := whale.NewSoftmaxCrossEntropy().Calculate(pred, t)
	if err != nil {
		panic(err)
	}

	loss.Backward()

	fmt.Println(loss)

	whale.VisualizeGraph(loss, "graph.png")
}
