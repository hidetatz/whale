package tensor

import (
	"fmt"
	"math/rand"
	"slices"
	"strings"
)

type Tensor struct {
	Data  []float64
	shape []int
}

// Scalar returns tensor as scalar.
func Scalar(s float64) *Tensor {
	return &Tensor{Data: []float64{s}}
}

// Vector returns tensor as vector.
func Vector(v []float64) *Tensor {
	return &Tensor{Data: v, shape: []int{len(v)}}
}

// Nd returns multi dimensional array.
// If the shape is empty, the given data is treated as vector.
func Nd(data []float64, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return Vector(data), nil
	}

	t := total(shape)
	if len(data) != t {
		return nil, fmt.Errorf("invalid shape, mismatch with data length")
	}

	return &Tensor{Data: data, shape: shape}, nil
}

// Rand creates a tensor by the given shape with randomized value [0.0, 1.0).
func Rand(shape ...int) *Tensor {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = rand.Float64()
	}
	t, _ := Nd(data, shape...) // error never happens
	return t
}

// Zeros creates a tensor by the given shape with all values 0.
func Zeros(shape ...int) *Tensor {
	data := make([]float64, total(shape)) // initialized by 0
	t, _ := Nd(data, shape...)            // error never happens
	return t
}

// ZerosLike creates a tensor by the given tensor's shape with all values 0.
func ZerosLike(t *Tensor) *Tensor {
	return Zeros(t.shape...)
}

// Ones creates a tensor by the given shape with all values 1.
func Ones(shape ...int) *Tensor {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = 1
	}
	t, _ := Nd(data, shape...) // error never happens
	return t
}

// OnesLike creates a tensor by the given tensor's shape with all values 1.
func OnesLike(t *Tensor) *Tensor {
	return Ones(t.shape...)
}

// All creates a tensor by the given shape with given value.
func All(v float64, shape ...int) *Tensor {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = v
	}
	t, _ := Nd(data, shape...) // error never happens
	return t
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

// Dim returns the dimension number.
func (t *Tensor) Dim() int {
	return len(t.shape)
}

// IsScalar returns true if the tensor is internally a scalar.
func (t *Tensor) IsScalar() bool {
	return len(t.shape) == 0
}

// IsVector returns true if the tensor is internally a vector.
func (t *Tensor) IsVector() bool {
	return len(t.shape) == 1
}

func toStrides(shape []int) []int {
	s := make([]int, len(shape))
	for i := range s {
		s[i] = total(shape[i+1:])
	}
	return s
}

func (t *Tensor) strides() []int {
	return toStrides(t.shape)
}

// String() implements Stringer interface.
func (t *Tensor) String() string {
	if t.IsScalar() {
		return fmt.Sprintf("%v", t.Data[0])
	}

	if t.IsVector() {
		return fmt.Sprintf("%v", t.Data)
	}

	var sb strings.Builder

	var w func(index []int)
	w = func(index []int) {
		indent := strings.Repeat("  ", len(index))

		if len(index) == len(t.shape)-1 {
			strides := t.strides()
			vars := []string{}
			for i := 0; i < t.shape[len(t.shape)-1]; i++ {
				idx := 0
				for j := range index {
					idx += index[j] * strides[j]
				}
				idx += i * strides[len(strides)-1]
				vars = append(vars, fmt.Sprintf("%v", t.Data[idx]))
			}
			sb.WriteString(fmt.Sprintf("%s[%s]\n", indent, strings.Join(vars, ", ")))
			return
		}

		sb.WriteString(fmt.Sprintf("%s[\n", indent))
		for i := 0; i < t.shape[len(index)]; i++ {
			w(append(index, i))
		}

		sb.WriteString(fmt.Sprintf("%s]\n", indent))
	}

	w([]int{})
	return sb.String()
}

// Equals retuens true is the 2 tensors are semantically the same.
// Even if they are on the different memory, if their data and shape are the same,
// it is treated as the same.
func (t *Tensor) Equals(t2 *Tensor) bool {
	return slices.Equal(t.shape, t2.shape) && slices.Equal(t.Data, t2.Data)
}

// Reshape returns an newly created tensor which has the same data, and the specified shape.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	t2 := t.Copy()
	return Nd(t2.Data, shape...)
}

// Copy creates a copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	ndata := make([]float64, len(t.Data))
	copy(ndata, t.Data)

	nshape := make([]int, len(t.shape))
	copy(nshape, t.shape)

	return &Tensor{Data: ndata, shape: nshape}
}

func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
	if len(axes) == 0 {
		// if empty, create [0, 1, 2...] slice and reverses it
		axes = seqi(0, len(t.shape))
		slices.Reverse(axes)
	}

	// check axes validity.
	// Let's say the dimension is 5,
	// axes must be the arbitrarily sorted slice of [0, 1, 2, 3, 4].

	// First check length
	if len(axes) != len(t.shape) {
		return nil, fmt.Errorf("invalid axes length: must be the same as the length of shape")
	}

	// Second, check if all the values are between 0 to dim.
	if slices.ContainsFunc(axes, func(n int) bool {
		if n < 0 || t.Dim() <= n {
			return true
		}
		return false
	}) {
		return nil, fmt.Errorf("invalid value in axes: must be in 0 to dim")
	}

	// Last, check if all the values are unique.
	copied := make([]int, len(axes))
	copy(copied, axes)
	slices.Sort(copied)
	copied = slices.Compact(copied)
	if len(axes) != len(copied) {
		return nil, fmt.Errorf("invalid value in axes: duplicate value contained")
	}

	// do transpose below

	newShape := make([]int, len(t.shape))
	for i := range axes {
		newShape[i] = t.shape[axes[i]]
	}

	newStrides := toStrides(newShape)

	curIndices := t.Indices()

	newData := make([]float64, len(t.Data))
	for _, curidx := range curIndices {
		newIdx := make([]int, len(t.shape))
		for i, axis := range axes {
			newIdx[i] += curidx.Idx[axis]
		}

		dataIdx := 0
		for i := range newIdx {
			dataIdx += newStrides[i] * newIdx[i]
		}
		newData[dataIdx] = curidx.Value
	}

	return &Tensor{Data: newData, shape: newShape}, nil
}

type Index struct {
	Idx   []int
	Value float64
}

func (t *Tensor) Indices() []*Index {
	strides := t.strides()

	indices := []*Index{}
	var f func(idx []int)
	f = func(idx []int) {
		if len(idx) == len(t.shape) {
			i := 0
			for j := range idx {
				i += idx[j] * strides[j]
			}
			indices = append(indices, &Index{Idx: idx, Value: t.Data[i]})
			return
		}

		for i := 0; i < t.shape[len(idx)]; i++ {
			f(append(idx, i))
		}
	}

	f([]int{})

	return indices
}

// Iterator returns the iterator of the tensor.
func (t *Tensor) Iterator(axis int) (*Iterator, error) {
	if axis > len(t.shape) {
		return nil, fmt.Errorf("axis mismatch")
	}

	return &Iterator{t: t, axis: axis}, nil
}

// Repeat copies the data on axis
func (t *Tensor) Repeat(times, axis int) (*Tensor, error) {
	ns := t.CopyShape()
	ns[axis] *= times

	nd := []float64{}

	iter, err := t.Iterator(axis)
	if err != nil {
		return nil, err
	}

	for iter.HasNext() {
		data := iter.Next()
		for i := 0; i < times; i++ {
			nd = append(nd, data...)
		}
	}

	return Nd(nd, ns...)
}

func (t *Tensor) Tile(times ...int) (*Tensor, error) {
	newshape := t.CopyShape()
	if len(t.shape) != len(times) {
		delta := len(times) - len(t.shape)
		for i := 0; i < delta; i++ {
			// push 1 to the head until the dim gets the same
			newshape = append([]int{1}, newshape...)
		}
	}

	nt, err := t.Reshape(newshape...)
	if err != nil {
		return nil, err
	}

	tmpshape := nt.CopyShape()
	tmpt := nt
	for axis, time := range times {
		tmpshape[axis] *= time
		tmpdata := []float64{}

		iter, err := tmpt.Iterator(axis - 1)
		if err != nil {
			return nil, err
		}

		for iter.HasNext() {
			data := iter.Next()
			for i := 0; i < time; i++ {
				tmpdata = append(tmpdata, data...)
			}
		}

		tt, err := Nd(tmpdata, tmpshape...)
		if err != nil {
			return nil, err
		}
		tmpt = tt
	}

	return tmpt, nil
}

func (t *Tensor) Sum(keepdims bool, axes ...int) (*Tensor, error) {
	if len(axes) == 0 {
		// when axes is empty, sum all.
		var result float64
		for i := range t.Data {
			result += t.Data[i]
		}

		if keepdims {
			shape := []int{}
			for i := 0; i < len(t.shape); i++ {
				shape = append(shape, 1)
			}

			return Nd([]float64{result}, shape...)
		}

		return Scalar(result), nil
	}

	// else, sum by axis

	curshape := t.CopyShape()

	slices.Sort(axes)
	slices.Reverse(axes) // ordered desc
	nt := t.Copy()
	strides := t.strides()
	for _, axis := range axes {
		axisdim := curshape[axis]

		datalen := total(nt.shape) / axisdim
		newdata := make([]float64, datalen)

		stride := strides[axis]
		took := []int{}
		for j := 0; j < datalen; j++ {
			left := 0
			for slices.Contains(took, left) {
				left++
			}

			result := 0.0
			for i := 0; i < axisdim; i++ {
				idx := left + i*stride
				result += nt.Data[idx]
				took = append(took, idx)
			}

			newdata[j] = result
		}

		if keepdims {
			// when keepdims is true, the calculated dimension will be 1.
			curshape[axis] = 1
		} else {
			// else, the calculated dimension is removed.
			curshape = append(curshape[:axis], curshape[axis+1:]...)
		}

		n, err := Nd(newdata, curshape...)
		if err != nil {
			return nil, err
		}
		nt = n
	}

	return nt, nil
}

func (t *Tensor) Squeeze(axes ...int) (*Tensor, error) {
	curshape := t.CopyShape()
	for _, axis := range axes {
		if curshape[axis] != 1 {
			return nil, fmt.Errorf("axis which is not 1 is specified")
		}
	}

	newshape := []int{}
	for i, dim := range curshape {
		if dim != 1 {
			newshape = append(newshape, dim)
			continue
		}

		if len(axes) != 0 && !slices.Contains(axes, i) {
			newshape = append(newshape, dim)
		}
	}

	return Nd(t.Data, newshape...)
}

func (t *Tensor) SumTo(shape ...int) (*Tensor, error) {
	ndim := len(shape)
	lead := t.Dim() - ndim
	leadAxis := seqi(0, lead)

	var axes []int
	for i, dim := range shape {
		if dim == 1 {
			axes = append(axes, i+lead)
		}
	}

	y, err := t.Sum(true, append(leadAxis, axes...)...)
	if err != nil {
		return nil, err
	}

	if lead > 0 {
		y2, err := y.Squeeze(leadAxis...)
		if err != nil {
			return nil, err
		}
		y = y2
	}

	return y, nil
}

func (t *Tensor) BroadcastTo(shape ...int) (*Tensor, error) {
	if len(t.shape) > len(shape) {
		return nil, fmt.Errorf("cannot broadcast: invalid shape")
	}

	newshape := make([]int, len(t.shape))
	copy(newshape, t.shape)
	if len(t.shape) != len(shape) {
		delta := len(shape) - len(t.shape)
		for i := 0; i < delta; i++ {
			// push 1 to the head until the dim gets the same
			newshape = append([]int{1}, newshape...)
		}
	}

	nt, err := t.Reshape(newshape...)
	if err != nil {
		return nil, err
	}

	tile := []int{}
	for i := range shape {
		if shape[i] == newshape[i] {
			tile = append(tile, 1)
			continue
		}

		if shape[i] != 1 && newshape[i] != 1 {
			return nil, fmt.Errorf("cannot broadcast: either dim must be 1 (original: %v, target: %v)", shape[i], newshape[i])
		}

		tile = append(tile, shape[i]/newshape[i])
	}

	nt, err = nt.Tile(tile...)
	if err != nil {
		return nil, err
	}

	return nt, nil
}

// Slice cuts the part of the tensor based on the given indices. The length of indices must be
// less than the tensor dimension.
// func (t *Tensor) Slice(s *Slice) (*tensor.Tensor, error) {
// 	if s == nil {
// 		return nil, fmt.Errorf("slice: nil input specified")
// 	}
//
// 	if len(s.index) > t.Dim() {
// 		return nil, fmt.Errorf("slice: too many indices specified, dim is %v but got %v", t.Dim(), len(s.index))
// 	}
//
// 	result := []float64{}
// 	for _, idx := range s.index {
//
// 	}
// }

// func (t *Tensor) SubTensor(idx int) *Tensor {
// 	ns := t.CopyShape()[1:]
// 	nst := t.CopyStrides()[1:]
// 	nd := []float64{}
//
// 	l := total(ns)
// 	start := t.strides[idx] * idx
// 	step := t.strides[idx+1]
// 	for i := 0; i < l; i++ {
// 		nd = append(nd, t.Data[start+i*step])
// 	}
//
// 	return &Tensor{Data: nd, shape: ns, strides: nst}
// }

func (t *Tensor) CopyShape() []int {
	ns := make([]int, len(t.shape))
	copy(ns, t.shape)
	return ns
}

func seq(from, to float64) []float64 {
	r := make([]float64, int(to-from))
	for i := from; i < to; i++ {
		r[int(i-from)] = float64(i)
	}
	return r
}

func seqi(from, to int) []int {
	r := make([]int, to-from)
	for i := from; i < to; i++ {
		r[i-from] = i
	}
	return r
}

func total(shape []int) int {
	total := 1
	for _, dim := range shape {
		total *= dim
	}
	return total
}
