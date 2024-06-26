package tensor

import (
	"fmt"
	"math/rand"
	"reflect"
)

/*
 * Factory functions which never causes error without whale itself bugs.
 */

// Scalar returns a tensor as scalar.
func Scalar(s float32) *Tensor { return &Tensor{data: []float32{s}} }

// Vector returns a tensor as vector.
func Vector(v []float32) *Tensor { return &Tensor{data: v, Shape: []int{len(v)}, Strides: []int{1}} }

// Rand creates a tensor by the given shape with randomized value [0.0, 1.0).
func Rand(shape ...int) *Tensor {
	data := make([]float32, product(shape))
	for i := range data {
		data[i] = rand.Float32()
	}
	return NdShape(data, shape...) // error never happens
}

// RandNorm creates a tensor by the given shape
// with values with distribution (mean = 0, stddev = 1).
func RandNorm(shape ...int) *Tensor {
	data := make([]float32, product(shape))
	for i := range data {
		data[i] = float32(rand.NormFloat64())
	}
	return NdShape(data, shape...) // error never happens
}

// Zeros creates a tensor by the given shape with all values 0.
func Zeros(shape ...int) *Tensor {
	data := make([]float32, product(shape)) // initialized by 0
	return NdShape(data, shape...)          // error never happens
}

// ZerosLike creates a tensor by the given tensor's shape with all values 0.
func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.Shape...)
}

// Ones creates a tensor by the given shape with all values 1.
func Ones(shape ...int) *Tensor {
	data := make([]float32, product(shape))
	for i := range data {
		data[i] = 1
	}
	return NdShape(data, shape...) // error never happens
}

// OnesLike creates a tensor by the given tensor's shape with all values 1.
func OnesLike(t *Tensor) *Tensor {
	return Ones(t.Shape...)
}

// Full creates a tensor by the given shape with given value.
func Full(v float32, shape ...int) *Tensor {
	data := make([]float32, product(shape))
	for i := range data {
		data[i] = v
	}
	return NdShape(data, shape...) // error never happens
}

// Arange creates a vector tensor by the given params.
func Arange(from, to, interval float32) *Tensor {
	data := make([]float32, int((to-from)/interval))
	for i := range data {
		data[i] = from + interval*float32(i)
	}

	return Vector(data)
}

// RandomPermutation creates a vector which has randomly shuffled from 0 to x.
func RandomPermutation(x int) *Tensor {
	perm := rand.Perm(x)
	r := make([]float32, len(perm))
	for i := range perm {
		r[i] = float32(perm[i])
	}

	return Vector(r)
}

/*
 * Factory functions which might cause error if arguments are invalid.
 */

// NdShape returns multi dimensional array by given data and shape.
// If the shape is empty, the given data is treated as vector.
func NdShape(data []float32, shape ...int) *Tensor { return MustGet(RespErr.NdShape(data, shape...)) }

// New inspects the dimension and shape of the given arr and creates a tensor based on them.
// The arr must be homogeneous, this consists of the 2 rules:
//   - every values must be the same type, currently float32.
//   - number of items on the same axis must be the same.
//
// See below for examples:
//   - Nd(2)                           -> returns Scalar(2)
//   - Nd([]float32{1, 2, 3})          -> returns Vector([1, 2, 3])
//   - Nd([][]float32{{1, 2}, {3, 4}}) -> returns NdShape([[1, 2, 3, 4], 2, 2) (= 2x2 matrix)
func New(arr any) *Tensor { return MustGet(RespErr.New(arr)) }

func (_ *plainErrResponser) NdShape(data []float32, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		if len(data) == 1 {
			return Scalar(data[0]), nil
		}
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

func (_ *plainErrResponser) New(arr any) (*Tensor, error) {
	val := reflect.ValueOf(arr)
	data := []float32{}
	shape := []int{}

	var f func(v reflect.Value, dim int) error
	f = func(v reflect.Value, dim int) error {
		if v.Kind() != reflect.Slice && v.Kind() != reflect.Float32 && v.Kind() != reflect.Float64 && v.Kind() != reflect.Int {
			return fmt.Errorf("array must be multi-dimensional slice of float32")
		}

		if v.Kind() == reflect.Int {
			data = append(data, float32(v.Int()))
			return nil
		}

		if v.Kind() == reflect.Float32 || v.Kind() == reflect.Float64 {
			data = append(data, float32(v.Float()))
			return nil
		}

		length := v.Len()

		if len(shape) == dim {
			shape = append(shape, length)
		} else {
			if length != shape[dim] {
				return fmt.Errorf("array must be homogeneous: %v", arr)
			}
		}

		for i := 0; i < v.Len(); i++ {
			if err := f(v.Index(i), dim+1); err != nil {
				return err
			}
		}

		return nil
	}
	if err := f(val, 0); err != nil {
		return nil, err
	}

	if len(shape) == 0 && len(data) == 1 {
		return Scalar(data[0]), nil
	}

	return RespErr.NdShape(data, shape...)
}
