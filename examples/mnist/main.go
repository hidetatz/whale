package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"runtime/pprof"

	"github.com/hidetatz/whale"
	"github.com/hidetatz/whale/tensor"
)

func main() {
	xi, ti, err := readMnistImages()
	if err != nil {
		panic(err)
	}

	xi.count = 1000
	ti.count = 1000

	xl, tl, err := readMnistLabels()
	if err != nil {
		panic(err)
	}

	xl.count = 1000
	tl.count = 1000

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

	train(xi, ti, xl, tl)

	// if xi.count != xl.count {
	// 	panic("unexpected count")
	// }

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
}

func preprocess(x []byte) *whale.Variable {
	xf := make([]float64, len(x))
	for i, b := range x {
		xf[i] = float64(int(b)) / 255.0
	}
	return whale.NewVar(tensor.MustNd(xf, 1, 784))
}

func train(xi, ti *MnistImage, xl, tl *MnistLabel) {
	layer := [][]int{{784, 100}, {100, 10}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(0.01))

	lossCalc := mlp.Loss()
	optim := mlp.Optimizer()

	for epoch := 0; epoch < 1; epoch++ {
		sumloss := 0.0
		for i := 0; i < xi.count; i++ {
			x := xi.pixels[i]
			t := xl.labels[i]

			xv := preprocess(x)
			y, err := mlp.Train(xv)
			if err != nil {
				panic(err)
			}

			tv := whale.NewVar(tensor.Vector([]float64{float64(t)}))
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

			sumloss += loss.GetData().Data[0] * float64(tv.Len())
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
		for i := 0; i < mi.count; i++ {
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
		for i := 0; i < ml.count; i++ {
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
