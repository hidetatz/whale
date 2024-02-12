package tensor2

import (
	"fmt"
	"math/rand"
)

// Scalar returns a tensor as scalar.
func Scalar(s float64) *Tensor { return &Tensor{data: []float64{s}} }

// Vector returns a tensor as vector.
func Vector(v []float64) *Tensor { return &Tensor{data: v, Shape: []int{len(v)}, Strides: []int{1}} }

// MustNd returns a multi dimensional tensor but panics on error.
func MustNd(data []float64, shape ...int) *Tensor {
	t, err := Nd(data, shape...)
	if err != nil {
		panic(err)
	}

	return t
}

// Nd returns multi dimensional array.
// If the shape is empty, the given data is treated as vector.
func Nd(data []float64, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return Vector(data), nil
	}

	if len(data) != product(shape) {
		return nil, fmt.Errorf("wrong shape: %v for %v data", shape, len(data))
	}

	strides := make([]int, len(shape))
	for i := range shape {
		strides[i] = product(shape[i+1:])
	}

	return &Tensor{data: data, Shape: shape, Strides: strides}, nil
}

// Rand creates a tensor by the given shape with randomized value [0.0, 1.0).
func Rand(shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = rand.Float64()
	}
	return MustNd(data, shape...) // error never happens
}

// RandNorm creates a tensor by the given shape
// with values with distribution (mean = 0, stddev = 1).
func RandNorm(shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return MustNd(data, shape...) // error never happens
}

// Zeros creates a tensor by the given shape with all values 0.
func Zeros(shape ...int) *Tensor {
	data := make([]float64, product(shape)) // initialized by 0
	return MustNd(data, shape...)           // error never happens
}

// ZerosLike creates a tensor by the given tensor's shape with all values 0.
func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.Shape...)
}

// Ones creates a tensor by the given shape with all values 1.
func Ones(shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = 1
	}
	return MustNd(data, shape...) // error never happens
}

// OnesLike creates a tensor by the given tensor's shape with all values 1.
func OnesLike(t *Tensor) *Tensor {
	return Ones(t.Shape...)
}

// Full creates a tensor by the given shape with given value.
func Full(v float64, shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = v
	}
	return MustNd(data, shape...) // error never happens
}

// Arange creates a tensor which has data between from and to by the given interval.
// If shape is not given, it is treated as vector.
// If from is bigger than to, the empty will be returned.
func Arange(from, to, interval float64, shape ...int) (*Tensor, error) {
	data := make([]float64, int((to-from)/interval))
	for i := range data {
		data[i] = from + interval*float64(i)
	}

	if len(shape) == 0 {
		return Vector(data), nil
	}

	t, err := Nd(data, shape...)
	if err != nil {
		return nil, err
	}

	return t, nil
}

// ArangeVec creates a vector tensor by the given params.
func ArangeVec(from, to, interval float64) *Tensor {
	data := make([]float64, int((to-from)/interval))
	for i := range data {
		data[i] = from + interval*float64(i)
	}

	return Vector(data)
}

// RandomPermutation creates a vector which has randomly shuffled from 0 to x.
func RandomPermutation(x int) *Tensor {
	perm := rand.Perm(x)
	r := make([]float64, len(perm))
	for i := range perm {
		r[i] = float64(perm[i])
	}

	return Vector(r)
}
