package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime/pprof"

	"github.com/hidetatz/whale"
	ts "github.com/hidetatz/whale/tensor"
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

	mnist, err := readMnist()
	if err != nil {
		panic(err)
	}

	layer := [][]int{{784, 1000}, {1000, 10}}
	mlp := whale.NewMLP(layer, true, whale.NewSigmoid(), whale.NewSoftmaxCrossEntropy(), whale.NewSGD(0.01))
	train(mlp, mnist.Train)

	// mlp.SaveGobFile("./mnist_mlp.gob")

	test := 1000
	correct := 0
	for i := range test {
		testdata := mnist.Test[i]
		lbl := testdata.label

		y := infer(mlp, []*MnistImage{testdata})
		inferResult := int(y.GetData().Argmax(false, -1).AsScalar())

		fmt.Printf("[%v] infer: %d, actual: %d\n", i, inferResult, lbl)

		if inferResult != lbl {
			// visualize read image
			testdata.Print()
		} else {
			correct++
		}
	}

	fmt.Printf("overall correctness: %v / %v\n", correct, test)
}

func infer(model whale.Model, data []*MnistImage) *whale.Variable {
	imgs := []float32{}
	for j := range data {
		imgs = append(imgs, data[j].pixels...)
	}

	x := whale.NewVar(ts.NdShape(imgs, len(data), len(data[0].pixels)))
	y, err := model.Train(x)
	if err != nil {
		panic(err)
	}

	return y
}

func train(model whale.Model, data []*MnistImage) {
	lossCalc := model.LossFn()
	optim := model.Optimizer()

	batch := 100
	ep := 1

	for epoch := 0; epoch < ep; epoch++ {
		var sumloss float32 = 0.0

		for i := range len(data) / batch {
			batchdata := data[i*batch : (i+1)*batch]
			y := infer(model, batchdata)

			lbls := []float32{}
			for j := range batchdata {
				lbls = append(lbls, float32(batchdata[j].label))
			}

			t := whale.NewVar(ts.Vector(lbls))

			loss, err := lossCalc.Calculate(y, t)
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

			sumloss += loss.GetData().AsScalar() * float32(t.Size())
			if i%100 == 0 {
				fmt.Println(epoch, ": ", i)
			}
		}

		fmt.Println("epoch: ", epoch+1, ", train loss: ", sumloss/float32(len(data)))
	}
}

type MnistImage struct {
	label  int
	pixels []float32
}

func (i *MnistImage) Print() {
	for i, pix := range i.pixels {
		if i%28 == 0 {
			fmt.Printf("\n")
		}
		if pix == 0 {
			fmt.Printf("  ")
		} else {
			fmt.Printf("XX")
		}
	}
	fmt.Printf("\n")
}

type Mnist struct {
	Train []*MnistImage
	Test  []*MnistImage
}

func readMnist() (*Mnist, error) {
	fn := func(imgfilename, labelfilename string, mnist *Mnist, train bool) error {
		readint32 := func(r io.Reader) int {
			b := make([]byte, 4)
			binary.Read(r, binary.BigEndian, &b)
			return int(binary.BigEndian.Uint32(b))
		}

		/*
		 * read image file
		 */

		imgf, err := os.Open(imgfilename)
		if err != nil {
			return err
		}
		defer imgf.Close()

		magic := readint32(imgf)
		if magic != 0x00000803 {
			return fmt.Errorf("unexpected magic in image file")
		}

		imgcnt := readint32(imgf)
		row := readint32(imgf)
		col := readint32(imgf)

		/*
		 * read label file
		 */

		lblf, err := os.Open(labelfilename)
		if err != nil {
			return err
		}
		defer lblf.Close()

		magic = readint32(lblf)
		if magic != 0x00000801 {
			return fmt.Errorf("unexpected magic in label file")
		}

		lblcnt := readint32(lblf)

		if imgcnt != lblcnt {
			return fmt.Errorf("differenc image count (%v) and label count (%v)", imgcnt, lblcnt)
		}

		images := make([]*MnistImage, imgcnt)

		for i := range imgcnt {
			images[i] = &MnistImage{}

			var lbl byte
			if err := binary.Read(lblf, binary.BigEndian, &lbl); err != nil {
				return err
			}

			images[i].label = int(lbl)

			pixels := make([]byte, row*col)
			if err = binary.Read(imgf, binary.BigEndian, &pixels); err != nil {
				return err
			}

			floats := make([]float32, len(pixels))
			for j, pixel := range pixels {
				floats[j] = float32(int(pixel)) / 255.0
			}

			images[i].pixels = floats
		}

		if train {
			mnist.Train = images
		} else {
			mnist.Test = images
		}

		return nil
	}

	mnist := &Mnist{}

	d := func(fname string) string {
		return fmt.Sprintf("./examples/mnist/%s", fname)
	}
	err := fn(d("train-images-idx3-ubyte"), d("train-labels-idx1-ubyte"), mnist, true)
	if err != nil {
		return nil, err
	}

	err = fn(d("t10k-images-idx3-ubyte"), d("t10k-labels-idx1-ubyte"), mnist, false)
	if err != nil {
		return nil, err
	}

	return mnist, nil
}
