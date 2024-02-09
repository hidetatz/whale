package main

import (
	"os"
	"encoding/binary"
	"fmt"
)

func main() {
	lbls, err := readMnistLabel()
	if err != nil {
		panic(err)
	}

	for _, lbl := range lbls {
		fmt.Println(lbl)
	}
}

type MnistLabel struct {
	count int
	labels []int
}

func readMnistLabel() ([]*MnistLabel, error) {
	files := []string{"train-labels-idx1-ubyte", "t10k-labels-idx1-ubyte"}

	labels := make([]*MnistLabel, 2)
	for i, file := range files {
		f, err := os.Open(fmt.Sprintf("./examples/mnist/%s", file))
		if err != nil {
			return nil, err
		}
		defer f.Close()

		var magic [4]byte
		err = binary.Read(f, binary.BigEndian, &magic)
		if err != nil {
			return nil, err
		}

		m := int(binary.BigEndian.Uint32(magic[:]))
		if m != 0x00000801 {
			return nil, fmt.Errorf("unexpected magic")
		}

		l := &MnistLabel{}

		var num [4]byte
		err = binary.Read(f, binary.BigEndian, &num)
		if err != nil {
			return nil, err
		}

		l.count = int(binary.BigEndian.Uint32(num[:]))
		l.labels = make([]int, l.count)

		for i := 0; i < l.count; i++ {
			var lbl [1]byte
			err = binary.Read(f, binary.BigEndian, &lbl)
			if err != nil {
				return nil, err
			}
			l.labels[i] = int(lbl[0])
		}
		labels[i] = l
	}

	return labels, nil
}
