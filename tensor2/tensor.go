package main

import (
	"fmt"
)

type Tensor struct {
	data []float32
}

func New(data []float32) *Tensor {
	return &Tensor{data: data}
}

func (t *Tensor) Materialize() []float32 {
	return t.data
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	data := make([]float32, len(t.data))
	for i := range t.data {
		data[i] = t.data[i] + t2.data[i]
	}
	return New(data)
}

func main() {
	t := New([]float32{1, 2})
	t2 := New([]float32{3, 4})
	t3 := t.Add(t2)
	fmt.Println(t3.Materialize())
}
