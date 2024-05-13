package tensor2

import (
	"fmt"
	"math/rand"
	"reflect"
)

/*
 * Factory functions which never causes error without whale itself bugs.
 */

// Scalar returns a tensor as scalar.
func Scalar(s float64) *Tensor { return &Tensor{data: []float64{s}} }

// Vector returns a tensor as vector.
func Vector(v []float64) *Tensor { return &Tensor{data: v, Shape: []int{len(v)}, Strides: []int{1}} }

// Rand creates a tensor by the given shape with randomized value [0.0, 1.0).
func Rand(shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = rand.Float64()
	}
	return NdShape(data, shape...) // error never happens
}

// RandNorm creates a tensor by the given shape
// with values with distribution (mean = 0, stddev = 1).
func RandNorm(shape ...int) *Tensor {
	data := make([]float64, product(shape))
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	return NdShape(data, shape...) // error never happens
}

// Zeros creates a tensor by the given shape with all values 0.
func Zeros(shape ...int) *Tensor {
	data := make([]float64, product(shape)) // initialized by 0
	return NdShape(data, shape...)          // error never happens
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
	return NdShape(data, shape...) // error never happens
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
	return NdShape(data, shape...) // error never happens
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

/*
 * Factory functions which might cause error if arguments are invalid.
 */

// NdShape returns multi dimensional array by given data and shape.
// If the shape is empty, the given data is treated as vector.
func NdShape(data []float64, shape ...int) *Tensor { return MustGet(RespErr.NdShape(data, shape...)) }

// New inspects the dimension and shape of the given arr and creates a tensor based on them.
// The arr must be homogeneous, this consists of the 2 rules:
//   - every values must be the same type, currently float64.
//   - number of items on the same axis must be the same.
//
// See below for examples:
//   - Nd(2)                           -> returns Scalar(2)
//   - Nd([]float64{1, 2, 3})          -> returns Vector([1, 2, 3])
//   - Nd([][]float64{{1, 2}, {3, 4}}) -> returns NdShape([[1, 2, 3, 4], 2, 2) (= 2x2 matrix)
func New(arr any) *Tensor { return MustGet(RespErr.New(arr)) }

// Arange creates a tensor which has data between from and to by the given interval.
// If shape is not given, it is treated as vector.
// If from is bigger than to, the empty will be returned.
func Arange(from, to, interval float64, shape ...int) *Tensor {
	return MustGet(RespErr.Arange(from, to, interval, shape...))
}

type errResponser struct{}

// RespErr makes it possible to handle error on library caller side.
// If a function/method is called via RespErr, error is returned if happened.
// If not, panic will be triggered on an error.
var RespErr = &errResponser{}

func (_ *errResponser) NdShape(data []float64, shape ...int) (*Tensor, error) {
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

func (_ *errResponser) New(arr any) (*Tensor, error) {
	val := reflect.ValueOf(arr)
	data := []float64{}
	shape := []int{}

	var f func(v reflect.Value, dim int) error
	f = func(v reflect.Value, dim int) error {
		if v.Kind() != reflect.Slice && v.Kind() != reflect.Float64 && v.Kind() != reflect.Int {
			return fmt.Errorf("array must be multi-dimensional slice of float64")
		}

		if v.Kind() == reflect.Int {
			data = append(data, float64(v.Int()))
			return nil
		}

		if v.Kind() == reflect.Float64 {
			data = append(data, v.Float())
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

func (_ *errResponser) Arange(from, to, interval float64, shape ...int) (*Tensor, error) {
	data := make([]float64, int((to-from)/interval))
	for i := range data {
		data[i] = from + interval*float64(i)
	}

	if len(shape) == 0 {
		return Vector(data), nil
	}

	return RespErr.NdShape(data, shape...)
}
