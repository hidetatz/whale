package whale

import (
	"fmt"
	"slices"
	"strings"
)

type Tensor struct {
	data    []float64
	shape   []int
	strides []int
}

func FromScalar(s float64) *Tensor {
	return &Tensor{data: []float64{s}}
}

func FromVector(v []float64, shape int) (*Tensor, error) {
	if len(v) != shape {
		return nil, fmt.Errorf("shape mismatch with vector length")
	}
	return &Tensor{data: v, shape: []int{shape}, strides: []int{1}}, nil
}

func Nd(data []float64, shape ...int) (*Tensor, error) {
	// scalar
	if len(shape) == 0 {
		if len(data) != 1 {
			return nil, fmt.Errorf("shape mismatch: scalar expected")
		}

		return FromScalar(data[0]), nil
	}

	// vector
	if len(shape) == 1 {
		return FromVector(data, shape[0])
	}

	// matrix/tensor
	elements := total(shape)
	if len(data) != elements {
		return nil, fmt.Errorf("shape mismatch with data length")
	}

	t := &Tensor{data: data, shape: shape}

	product := func(a []int) int {
		r := 1
		for i := range a {
			r *= a[i]
		}
		return r
	}

	for i := range shape {
		t.strides = append(t.strides, product(shape[i+1:]))
	}

	return t, nil
}

func Zeros(shape ...int) (*Tensor, error) {
	data := make([]float64, total(shape)) // initialized by 0
	return Nd(data, shape...)
}

func Ones(shape ...int) (*Tensor, error) {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = 1
	}
	return Nd(data, shape...)
}

func seq(from, to int) []float64 {
	r := make([]float64, to-from)
	for i := from; i < to; i++ {
		r[i-from] = float64(i)
	}
	return r
}

func ArangeTo(to int) (*Tensor, error) {
	if to < 0 {
		return nil, fmt.Errorf("arg should be positive")
	}

	data := seq(0, to)
	return FromVector(data, len(data))
}

func ArangeFrom(from, to int) (*Tensor, error) {
	if to < from {
		return nil, fmt.Errorf("from should not be bigger than to")
	}

	data := seq(from, to)
	return FromVector(data, len(data))
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Strides() []int {
	return t.strides
}

func (t *Tensor) Dim() int {
	return len(t.shape)
}

func (t *Tensor) String() string {
	var sb strings.Builder

	// When the tensor is N-dimension array,
	// The tensor[a][b][c]... (this lasts N times) will be:
	// a * strides[0] + b * strides[1] + c * strides[2] + ... (this lasts N times).
	// The argument is a slice like [a] or [a, b] or [a, b, c] or ...
	var w func(index []int)
	w = func(index []int) {
		indent := strings.Repeat("  ", len(index))

		// If length of index == N - 1, then comes here.
		// This is a special case because it needs to print actual value without indentation.
		if len(index) == t.Dim()-1 {
			laststride := t.strides[len(t.strides)-1]

			sb.WriteString(fmt.Sprintf("%s[", indent))
			for i := 0; i < t.shape[t.Dim()-1]; i++ {
				// Do a * strides[0] + b * strides[1]...
				idx := 0
				for j := range index {
					idx += index[j] * t.strides[j]
				}

				// Add the last stride.
				idx += i * laststride

				if i > 0 {
					sb.WriteString(", ")
				}
				sb.WriteString(fmt.Sprintf("%v", t.data[idx]))
			}
			sb.WriteString("]\n")

			return
		}

		// If length of index is smaller then N - 1,
		// Append the index and do recursive.
		sb.WriteString(fmt.Sprintf("%s[\n", indent))
		length := t.shape[len(index)]
		for i := 0; i < length; i++ {
			nindex := make([]int, len(index))
			copy(nindex, index)
			nindex = append(nindex, i)
			w(nindex)
		}

		sb.WriteString(fmt.Sprintf("%s]\n", indent))
	}

	w([]int{})
	return sb.String()
}

func (t *Tensor) Equals(t2 *Tensor) bool {
	if !slices.Equal(t.shape, t2.shape) {
		return false
	}

	if !slices.Equal(t.strides, t2.strides) {
		return false
	}

	if !slices.Equal(t.data, t2.data) {
		return false
	}

	return true
}

func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if total(t.shape) != total(shape) {
		return nil, fmt.Errorf("cannot reshape: the data size mismatch")
	}

	return Nd(t.data, shape...)
}

func (t *Tensor) Copy() *Tensor {
	return &Tensor{
		data:    t.data,
		shape:   t.shape,
		strides: t.strides,
	}
}

func (t *Tensor) Transpose() *Tensor {
	nt := t.Copy()
	slices.Reverse(nt.shape)
	slices.Reverse(nt.strides)
	return nt
}

func (t *Tensor) TransposeAxes(axes ...int) (*Tensor, error) {
	// check axes validity.
	// Let's say the dimension is 5,
	// axes must be the arbitrarily sorted slice of [0, 1, 2, 3, 4].

	// First check length
	if len(axes) != len(t.shape) {
		return nil, fmt.Errorf("length of axes does not match the shape")
	}

	// Second, check if all the values are between 0 to dim.
	if slices.ContainsFunc(axes, func(n int) bool {
		if n < 0 || t.Dim() <= n {
			return true
		}
		return false
	}) {
		return nil, fmt.Errorf("invalid value in axes: must be in 0 to dimension")
	}

	// Last, check if all the values are unique.
	copied := make([]int, len(axes))
	copy(copied, axes)
	slices.Sort(copied)
	copied = slices.Compact(copied)
	if len(axes) != len(copied) {
		fmt.Println(axes, copied)
		return nil, fmt.Errorf("duplicate value in axes")
	}

	nt := t.Copy()
	newshape := make([]int, len(nt.shape))
	newstrides := make([]int, len(nt.strides))
	for i := range axes {
		newshape[i] = t.shape[axes[i]]
		newstrides[i] = t.strides[axes[i]]
	}

	nt.shape = newshape
	nt.strides = newstrides
	return nt, nil
}

//	func (t *Tensor) BroadcastTo(shape ...int) (*Tensor, error) {
//		if total(t.shape) > total(shape) {
//			return nil, fmt.Errorf("cannot broadcast: ndim must be smaller then the given one")
//		}
//
//		newshape := make([]int, len(t.shape))
//		copy(newshape, t.shape)
//		if len(t.shape) != len(shape) {
//			delta := len(shape) - len(newshape)
//			for i := 0; i < delta; i++ {
//				// push 1 to the head until the dim gets the same
//				newshape = append([]int{1}, newshape...)
//			}
//		}
//
//		nt, err := t.Reshape(newshape...)
//		if err != nil {
//			return nil, err
//		}
//		fmt.Println(nt, nt.data, nt.shape, nt.Dim(), shape)
//		nt, err = nt.Tile(shape...)
//		if err != nil {
//			return nil, err
//		}
//
//		return nt, nil
//	}

func total(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}

// func (t *Tensor) Tile(newShape ...int) (*Tensor, error) {
// 	newSize := total(newShape)
// 	newData := make([]float64, newSize)
//
// 	broadcastIndex := func(idx int, shape []int) int {
// 		var result int
// 		for i, s := range shape {
// 			result += (idx / total(shape[i+1:])) % s * total(newShape[i+1:])
// 		}
// 		return result
// 	}
//
// 	for i := 0; i < newSize; i++ {
// 		sourceIndex := broadcastIndex(i, newShape)
// 		newData[i] = t.data[sourceIndex]
// 	}
//
// 	// for i := 0; i < len(newShape); i++ {
// 	// 	if newShape[i] == t.shape[i] {
// 	// 		continue
// 	// 	}
//
// 	// 	if t.shape[i] == 1 {
// 	// 		for j := 0; j < t.shape[i]; j++ {
// 	// 			copy(newData[j*newSize:(j+1)*newSize], t.data[j*total(t.shape[i+1:]):(j+1)*total(t.shape[i+1:])])
// 	// 		}
//
// 	// 		continue
// 	// 	}
//
// 	// 	return nil, fmt.Errorf("cannot tile: the length of %dd is %d, but 1 is expected.", i, newShape[i])
// 	// }
//
// 	return &Tensor{data: newData, shape: newShape, dim: len(newShape)}, nil
// }
