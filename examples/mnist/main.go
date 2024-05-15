package main

import (
	"encoding/binary"
	"fmt"
	"os"

	"github.com/hidetatz/whale"
	ts "github.com/hidetatz/whale/tensor2"
)

func main() {
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

	train(xi, ti, xl, tl)
}

func preprocess(x []byte) *whale.Variable {
	xf := make([]float64, len(x))
	for i, b := range x {
		xf[i] = float64(int(b)) / 255.0
	}
	return whale.NewVar(ts.NdShape(xf, 1, 784))
}

func train(xi, ti *MnistImage, xl, tl *MnistLabel) {
	layer := [][]int{{784, 100}, {100, 10}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(0.01))

	lossCalc := mlp.Loss()
	optim := mlp.Optimizer()

	for epoch := 0; epoch < 5; epoch++ {
		sumloss := 0.0
		for i := range xi.count {
			x := xi.pixels[i]
			t := xl.labels[i]

			xv := preprocess(x)
			y, err := mlp.Train(xv)
			if err != nil {
				panic(err)
			}

			tv := whale.NewVar(ts.Vector([]float64{float64(t)}))
			loss, err := lossCalc.Calculate(y, tv)
			if err != nil {
				panic(err)
			}

			params := mlp.Params()
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

		fmt.Println("epoch: ", epoch+1, ", train loss: ", sumloss/float64(ti.count))
	}
}

type MnistImage struct {
	count  int
	pixels [][]byte
}

type MnistLabel struct {
	count  int
	labels []int
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
		mi.pixels = make([][]byte, mi.count)

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

			mi.pixels[i] = pixel
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
		ml.labels = make([]int, ml.count)

		// read actual label
		for i := range ml.count {
			var lbl byte
			if err = binary.Read(f, binary.BigEndian, &lbl); err != nil {
				return nil, err
			}

			ml.labels[i] = int(lbl)
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
