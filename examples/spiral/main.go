package main

import (
	"fmt"
	"math"

	"github.com/hidetatz/whale"
	tensor "github.com/hidetatz/whale/tensor2"
)

var p = whale.NewPlot()

func main() {
	// spiral test data
	xt, tt := whale.RandSpiral()
	plot(xt, tt)

	x := whale.NewVar(xt)
	t := whale.NewVar(tt)

	// training
	layer := [][]int{{2, 10}, {10, 3}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(1.0))
	Train(mlp, x, t)

	// visualize trained result

	// find test data min and max point
	xmin := math.Inf(1)
	ymin := math.Inf(1)
	xmax := math.Inf(-1)
	ymax := math.Inf(-1)
	for i, d := range xt.Flatten() {
		if i%2 == 0 {
			if d < xmin {
				xmin = d
			} else if xmax < d {
				xmax = d
			}
		} else {
			if d < ymin {
				ymin = d
			} else if ymax < d {
				ymax = d
			}
		}
	}

	X, Y, err := tensor.MeshGrid(tensor.ArangeVec(xmin, xmax, 0.01), tensor.ArangeVec(ymin, ymax, 0.01))
	if err != nil {
		panic(err)
	}

	// interleave
	xdata := []float64{}
	xf := X.Flatten()
	yf := Y.Flatten()
	for i := range xf {
		xdata = append(xdata, xf[i])
		xdata = append(xdata, yf[i])
	}

	xd := tensor.Must(tensor.NdShape(xdata, len(xf), 2))

	score, err := mlp.Train(whale.NewVar(xd))
	if err != nil {
		panic(err)
	}

	z, err := score.GetData().Argmax(false, 1)
	if err != nil {
		panic(err)
	}

	// z, err = z.Reshape(xx.Shape...)
	// if err != nil {
	// 	panic(err)
	// }

	xs0 := []float64{}
	ys0 := []float64{}

	xs1 := []float64{}
	ys1 := []float64{}

	xs2 := []float64{}
	ys2 := []float64{}

	zdata := z.Flatten()
	for i := 0; i < len(xdata); i += 2 {
		label := zdata[i/2]
		switch label {
		case 0:
			xs0 = append(xs0, xdata[i])
			ys0 = append(ys0, xdata[i+1])
		case 1:
			xs1 = append(xs1, xdata[i])
			ys1 = append(ys1, xdata[i+1])
		case 2:
			xs2 = append(xs2, xdata[i])
			ys2 = append(ys2, xdata[i+1])
		}
	}

	if err := p.Scatter(xs0, ys0, "blue"); err != nil {
		panic(err)
	}

	if err := p.Scatter(xs1, ys1, "red"); err != nil {
		panic(err)
	}

	if err := p.Scatter(xs2, ys2, "green"); err != nil {
		panic(err)
	}

	if err := p.Exec(); err != nil {
		panic(err)
	}
}

func Train(m whale.Model, x, t *whale.Variable) {
	lossCalc := m.Loss()
	optim := m.Optimizer()

	batch := 30

	for epoch := 0; epoch < 300; epoch++ {
		index := tensor.RandomPermutation(300)
		sumloss := 0.0

		for i := range 10 {
			batchIdx, err := index.Index(tensor.FromTo(i*batch, (i+1)*batch))
			if err != nil {
				panic(err)
			}

			batchX, err := whale.Index(x, whale.NewVar(batchIdx))
			if err != nil {
				panic(err)
			}

			batchT, err := whale.Index(t, whale.NewVar(batchIdx))
			if err != nil {
				panic(err)
			}

			y, err := m.Train(batchX)
			if err != nil {
				panic(err)
			}

			loss, err := lossCalc.Calculate(y, batchT)
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

			sumloss += loss.GetData().AsScalar() * float64(batchT.Len())
		}

		fmt.Println("epoch: ", epoch+1, ", loss: ", sumloss/300)
	}
}

func plot(x, t *tensor.Tensor) {
	xs0 := []float64{}
	ys0 := []float64{}

	xs1 := []float64{}
	ys1 := []float64{}

	xs2 := []float64{}
	ys2 := []float64{}

	for i := range t.Size() {
		// 0 / 1 / 2
		label := tensor.Must(t.Index(tensor.At(i))).AsScalar()

		st, err := x.Index(tensor.At(i))
		if err != nil {
			panic(err)
		}

		data := st.Flatten()
		switch label {
		case 0:
			xs0 = append(xs0, data[0])
			ys0 = append(ys0, data[1])
		case 1:
			xs1 = append(xs1, data[0])
			ys1 = append(ys1, data[1])
		case 2:
			xs2 = append(xs2, data[0])
			ys2 = append(ys2, data[1])
		}
	}

	if err := p.Scatter(xs0, ys0, "blue"); err != nil {
		panic(err)
	}

	if err := p.Scatter(xs1, ys1, "red"); err != nil {
		panic(err)
	}

	if err := p.Scatter(xs2, ys2, "green"); err != nil {
		panic(err)
	}
}
