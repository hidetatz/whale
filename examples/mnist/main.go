package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime/pprof"
	"slices"

	"github.com/hidetatz/whale"
	ts "github.com/hidetatz/whale/tensor2"
)

func main() {
	f, err := os.Create("cpu.prof")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	if err := pprof.StartCPUProfile(f); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	xi, ti, err := readMnistImages()
	if err != nil {
		panic(err)
	}

	xl, tl, err := readMnistLabels()
	if err != nil {
		panic(err)
	}

	// visualize read image
	// for i := range xi.pixels {
	// 	lbl := xl.labels[i]
	// 	fmt.Printf("%v: ", lbl)
	// 	for i, pix := range xi.pixels[i] {
	// 		if i % 28 == 0 {
	// 			fmt.Printf("\n")
	// 		}
	// 		fmt.Printf("%v%s", pix, strings.Repeat(" ", 3 - digits(int(pix))))
	// 	}
	// 	fmt.Printf("\n\n")
	// }

	layer := [][]int{{784, 1000}, {1000, 10}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(0.01))
	train(mlp, xi, xl)

	mlp.SaveGobFile("./mnist_mlp.gob")
}

func train(model whale.Model, xi *MnistImage, xl *MnistLabel) {
	lossCalc := model.LossFn()
	optim := model.Optimizer()

	batch := 100

	for epoch := 0; epoch < 5; epoch++ {
		sumloss := 0.0

		for i := range xi.count / batch {
			imgs := xi.pixels[i*batch : (i+1)*batch]
			lbls := xl.labels[i*batch : (i+1)*batch]

			x := slices.Concat(imgs...)
			xv := whale.NewVar(ts.NdShape(x, batch, 28*28))

			y, err := model.Train(xv)
			if err != nil {
				panic(err)
			}

			tv := whale.NewVar(ts.Vector(lbls))

			loss, err := lossCalc.Calculate(y, tv)
			if err != nil {
				panic(err)
			}

			params := model.Params()
			for _, p := range params {
				p.ClearGrad()
			}

			loss.Backward()
			for _, p := range params {
				optim.Optimize(p)
			}

			sumloss += loss.GetData().AsScalar() * float64(tv.Size())
			if i%100 == 0 {
				fmt.Println(epoch, ": ", i)
			}
		}

		fmt.Println("epoch: ", epoch+1, ", train loss: ", sumloss/float64(xi.count), sumloss, xi.count)
	}
}

type MnistImage struct {
	count  int
	pixels [][]float64
}

type MnistLabel struct {
	count  int
	labels []float64
}

func digits(i int) int {
	if i == 0 {
		return 1
	}
	count := 0
	for i != 0 {
		i /= 10
		count++
	}
	return count
}

func readMnistImages() (x, t *MnistImage, err error) {
	fn := func(filename string) (*MnistImage, error) {
		f, err := os.Open(fmt.Sprintf("./examples/mnist/%s", filename))
		if err != nil {
			return nil, err
		}
		defer f.Close()

		// read magic number
		magic := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &magic); err != nil {
			return nil, err
		}

		if int(binary.BigEndian.Uint32(magic)) != 0x00000803 {
			return nil, fmt.Errorf("unexpected magic")
		}

		mi := &MnistImage{}

		// read data count
		num := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &num); err != nil {
			return nil, err
		}

		mi.count = int(binary.BigEndian.Uint32(num))
		mi.pixels = make([][]float64, mi.count)

		// read rows and cols
		rows := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &rows); err != nil {
			return nil, err
		}

		r := int(binary.BigEndian.Uint32(rows))

		cols := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &cols); err != nil {
			return nil, err
		}

		c := int(binary.BigEndian.Uint32(cols))

		// read actual data
		for i := range mi.count {
			pixel := make([]byte, r*c)
			if err = binary.Read(f, binary.BigEndian, &pixel); err != nil {
				return nil, err
			}

			fs := make([]float64, len(pixel))
			for j, b := range pixel {
				fs[j] = float64(int(b)) / 255.0
			}

			mi.pixels[i] = fs
		}

		return mi, nil
	}

	x, err = fn("train-images-idx3-ubyte")
	if err != nil {
		return nil, nil, err
	}

	t, err = fn("t10k-images-idx3-ubyte")
	if err != nil {
		return nil, nil, err
	}

	return x, t, nil
}

func readMnistLabels() (x, t *MnistLabel, err error) {
	fn := func(filename string) (*MnistLabel, error) {
		f, err := os.Open(fmt.Sprintf("./examples/mnist/%s", filename))
		if err != nil {
			return nil, err
		}
		defer f.Close()

		// read magic number
		magic := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &magic); err != nil {
			return nil, err
		}

		if int(binary.BigEndian.Uint32(magic)) != 0x00000801 {
			return nil, fmt.Errorf("unexpected magic")
		}

		ml := &MnistLabel{}

		// read data count
		num := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &num); err != nil {
			return nil, err
		}

		ml.count = int(binary.BigEndian.Uint32(num))
		ml.labels = make([]float64, ml.count)

		// read actual label
		for i := range ml.count {
			var lbl byte
			if err = binary.Read(f, binary.BigEndian, &lbl); err != nil {
				return nil, err
			}

			ml.labels[i] = float64(int(lbl))
		}

		return ml, nil
	}

	x, err = fn("train-labels-idx1-ubyte")
	if err != nil {
		return nil, nil, err
	}

	t, err = fn("t10k-labels-idx1-ubyte")
	if err != nil {
		return nil, nil, err
	}

	return x, t, nil
}
