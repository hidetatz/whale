package main

import (
	"os"
	"encoding/binary"
	"fmt"
	"strings"
)

func main() {
	xi, _, err := readMnistImages()
	if err != nil {
		panic(err)
	}

	xl, _, err := readMnistLabels()
	if err != nil {
		panic(err)
	}

	if xi.count != xl.count {
		panic("unexpected count")
	}

	for i := range xi.pixels {
		lbl := xl.labels[i]

		fmt.Printf("%v: ", lbl)
		for i, pix := range xi.pixels[i] {
			if i % 28 == 0 {
				fmt.Printf("\n")
			}
			fmt.Printf("%v%s", pix, strings.Repeat(" ", 3 - digits(int(pix))))
		}

		fmt.Printf("\n\n")
	}
}

type MnistImage struct {
	count int
	pixels [][]byte
}

type MnistLabel struct {
	count int
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

		magic := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &magic); err != nil {
			return nil, err
		}

		if int(binary.BigEndian.Uint32(magic)) != 0x00000803 {
			return nil, fmt.Errorf("unexpected magic")
		}

		mi := &MnistImage{}

		num := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &num); err != nil {
			return nil, err
		}

		mi.count = int(binary.BigEndian.Uint32(num))
		mi.pixels = make([][]byte, mi.count)

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

		for i := 0; i < mi.count; i++ {
			pixel := make([]byte, r * c)
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

		magic := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &magic); err != nil {
			return nil, err
		}

		if int(binary.BigEndian.Uint32(magic)) != 0x00000801 {
			return nil, fmt.Errorf("unexpected magic")
		}

		ml := &MnistLabel{}

		num := make([]byte, 4)
		if err = binary.Read(f, binary.BigEndian, &num); err != nil {
			return nil, err
		}

		ml.count = int(binary.BigEndian.Uint32(num))
		ml.labels = make([]int, ml.count)

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
