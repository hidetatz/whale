package main

import (
	"fmt"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

var p = whale.NewPlot()

func main() {
	// spiral test data
	xt, tt := whale.RandSpiral()

	x := whale.NewVar(xt)
	t := whale.NewVar(tt)

	/*
	 * do training
	 */
	layer := [][]int{{2, 10}, {10, 3}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(1.0))
	Train(mlp, x, t)

	/*
	 * inference and visualize trained result
	 */

	// testdata
	xmin := x.Index(tensor.All(), tensor.At(0)).Min().AsScalar() - 0.1
	xmax := x.Index(tensor.All(), tensor.At(0)).Max().AsScalar() + 0.1

	ymin := x.Index(tensor.All(), tensor.At(1)).Min().AsScalar() - 0.1
	ymax := x.Index(tensor.All(), tensor.At(1)).Max().AsScalar() + 0.1

	// test date with even intervals.
	// x-axis-size = (xmax - xmin) / 0.05
	// y-axis-size = (xmax - xmin) / 0.05
	xs := tensor.Arange(xmin, xmax, 0.05)
	ys := tensor.Arange(ymin, ymax, 0.05)

	// X, Y's shape: (x-axis-size, y-axis-size)
	X, Y := tensor.MeshGrid(xs, ys)

	// convert into the shape (x-axis-size * y-axis-size, 2) to input into the model.
	// [
	//   [x_0, y_0],
	//   [x_1, y_1],
	//   [x_2, y_2],
	//   ...
	//   [x_x-axis-size*y-axis-size, y_x-axis-size*y-axis-size],
	// ]
	// total size is (x-axis-size * y-axis-size * 2).
	testdata := []float64{}
	xf := X.Flatten()
	yf := Y.Flatten()
	for i := range xf {
		testdata = append(testdata, xf[i])
		testdata = append(testdata, yf[i])
	}
	xd := tensor.NdShape(testdata, X.Size(), 2)

	// do inference.
	// score's shape is (x-axis-size * y-axis-size, 3)
	// [
	//   // vertical size is (x-axis-size * y-axis-size)
	//   [p1, p2, p3],
	//   [p1, p2, p3],
	//   [p1, p2, p3],
	//   ...
	//   [p1, p2, p3],
	// ]
	// total size is (x-axis-size * y-axis-size * 2).
	score, err := mlp.Train(whale.NewVar(xd))
	if err != nil {
		panic(err)
	}

	// pick max from (p1, p2, p3)
	// shape: (x-axis-size * y-axis-size)
	// [
	//   // vertical size is (x-axis-size * y-axis-size)
	//   0,
	//   2,
	//   2,
	//   1,
	//   0,
	//   ...
	//   1,
	// ]
	prediceCls := score.GetData().Argmax(false, 1)

	xs0 := []float64{}
	ys0 := []float64{}

	xs1 := []float64{}
	ys1 := []float64{}

	xs2 := []float64{}
	ys2 := []float64{}

	pi := prediceCls.Iterator()
	for pi.HasNext() {
		i, label := pi.Next()
		switch label {
		case 0:
			xs0 = append(xs0, xf[i])
			ys0 = append(ys0, yf[i])
		case 1:
			xs1 = append(xs1, xf[i])
			ys1 = append(ys1, yf[i])
		case 2:
			xs2 = append(xs2, xf[i])
			ys2 = append(ys2, yf[i])
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

	plot(xt, tt)

	if err := p.Exec(); err != nil {
		panic(err)
	}

}

func Train(m whale.Model, x, t *whale.Variable) {
	lossCalc := m.LossFn()
	optim := m.Optimizer()

	batch := 30

	for epoch := 0; epoch < 300; epoch++ {
		index := tensor.RandomPermutation(300)
		sumloss := 0.0

		for i := range 10 {
			batchIdx := index.Index(tensor.FromTo(i*batch, (i+1)*batch))

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

			sumloss += loss.GetData().AsScalar() * float64(batchT.Size())
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
		label := t.Index(tensor.At(i)).AsScalar()

		st := x.Index(tensor.At(i))

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
