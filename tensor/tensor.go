package tensor

import (
	"fmt"
	"math/rand"
	"slices"
	"strings"
)

type Tensor struct {
	Data  []float64
	Shape []int
}

// Empty returns empty tensor.
func Empty() *Tensor {
	return &Tensor{}
}

// Scalar returns tensor as scalar.
func Scalar(s float64) *Tensor {
	return &Tensor{Data: []float64{s}}
}

// Vector returns tensor as vector.
func Vector(v []float64) *Tensor {
	return &Tensor{Data: v, Shape: []int{len(v)}}
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

	return &Tensor{Data: data, Shape: shape}, nil
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
	return Zeros(t.Shape...)
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
	return Ones(t.Shape...)
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
	return len(t.Shape)
}

// IsScalar returns true if the tensor is internally a scalar.
func (t *Tensor) IsScalar() bool {
	return len(t.Shape) == 0
}

// IsVector returns true if the tensor is internally a vector.
func (t *Tensor) IsVector() bool {
	return len(t.Shape) == 1
}

// Strides returns the stride of the tensor.
// Stride is the intervel count between the next dimension.
func (t *Tensor) Strides() []int {
	return toStrides(t.Shape)
}

// String implements Stringer interface.
func (t *Tensor) String() string {
	if len(t.Data) == 0 {
		return fmt.Sprintf("[]")
	}

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

		if len(index) == len(t.Shape)-1 {
			strides := t.Strides()
			vars := []string{}
			for i := 0; i < t.Shape[len(t.Shape)-1]; i++ {
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
		for i := 0; i < t.Shape[len(index)]; i++ {
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
	return slices.Equal(t.Shape, t2.Shape) && slices.Equal(t.Data, t2.Data)
}

// Reshape returns an newly created tensor which has the same data, and the specified shape.
func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	return Nd(t.Data, shape...)
}

// Copy creates a copy of the tensor.
func (t *Tensor) Copy() *Tensor {
	ndata := make([]float64, len(t.Data))
	copy(ndata, t.Data)

	nshape := make([]int, len(t.Shape))
	copy(nshape, t.Shape)

	return &Tensor{Data: ndata, Shape: nshape}
}

// Transpose transposes the tensor by the given axis.
// If empty is given, all the axes are reversed.
func (t *Tensor) Transpose(axes ...int) (*Tensor, error) {
	if t.IsScalar() {
		return t.Copy(), nil
	}

	if len(axes) == 0 {
		// if empty, create [0, 1, 2...] slice and reverses it
		axes = seqi(0, len(t.Shape))
		slices.Reverse(axes)
	}

	// check axes validity.
	// Let's say the dimension is 5,
	// axes must be the arbitrarily sorted slice of [0, 1, 2, 3, 4].

	// First check length
	if len(axes) != len(t.Shape) {
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

	// do transpose

	newShape := make([]int, len(t.Shape))
	for i := range axes {
		newShape[i] = t.Shape[axes[i]]
	}

	newStrides := toStrides(newShape)

	curIndices := t.Indices()

	newData := make([]float64, len(t.Data))
	for _, curidx := range curIndices {
		newIdx := make([]int, len(t.Shape))
		for i, axis := range axes {
			newIdx[i] += curidx.Idx[axis]
		}

		dataIdx := 0
		for i := range newIdx {
			dataIdx += newStrides[i] * newIdx[i]
		}
		newData[dataIdx] = curidx.Value
	}

	return &Tensor{Data: newData, Shape: newShape}, nil
}

type Index struct {
	Idx   []int
	Value float64
}

func (i *Index) String() string {
	return fmt.Sprintf("{%v: %v}", i.Idx, i.Value)
}

// Indices returns every value's index in the tensor.
func (t *Tensor) Indices() []*Index {
	strides := t.Strides()

	indices := []*Index{}
	var f func(idx []int)
	f = func(idx []int) {
		if len(idx) == len(t.Shape) {
			c := copySlice(idx)
			i := 0
			for j := range c {
				i += c[j] * strides[j]
			}
			indices = append(indices, &Index{Idx: c, Value: t.Data[i]})
			return
		}

		for i := 0; i < t.Shape[len(idx)]; i++ {
			f(append(idx, i))
		}
	}

	f([]int{})

	return indices
}

// Iterator returns the iterator of the tensor.
func (t *Tensor) Iterator(axis int) (*Iterator, error) {
	if axis > len(t.Shape) {
		return nil, fmt.Errorf("axis mismatch")
	}

	return &Iterator{t: t, axis: axis}, nil
}

// SubTensor returns the part of the tensor based on the given index.
// Returned tensor is newly created one and does not have connection to the origin.
func (t *Tensor) SubTensor(index []int) (*Tensor, error) {
	if len(index) > len(t.Shape) {
		return nil, fmt.Errorf("too many index specified")
	}

	curStride := t.Strides()
	newShape := t.CopyShape()[len(index):]
	length := total(newShape)

	newData := make([]float64, length)

	start := 0
	for i := range index {
		if index[i] > t.Shape[i]-1 {
			return nil, fmt.Errorf("index is too big: %v", index)
		}
		start += curStride[i] * index[i]
	}
	for i := 0; i < length; i++ {
		newData[i] = t.Data[start+i]
	}
	return Nd(newData, newShape...)
}

// Repeat copies the data on axis.
// func (t *Tensor) Repeat(repeats []int, axis int) (*Tensor, error) {
// 	ns := t.CopyShape()
// 	ns[axis] *= times
//
// 	nd := []float64{}
//
// 	iter, err := t.Iterator(axis)
// 	if err != nil {
// 		return nil, err
// 	}
//
// 	for iter.HasNext() {
// 		data := iter.Next()
// 		for i := 0; i < times; i++ {
// 			nd = append(nd, data...)
// 		}
// 	}
//
// 	return Nd(nd, ns...)
// }

func (t *Tensor) genIndices(dim int) [][]int {
	var indices [][]int
	index := make([]int, dim)

	var generate func(int)
	generate = func(d int) {
		if d == dim {
			indices = append(indices, append([]int{}, index...))
			return
		}

		for i := 0; i < t.Shape[d]; i++ {
			index[d] = i
			generate(d + 1)
		}
	}

	generate(0)
	return indices
}

// Tile repeats the tensor by the given reps like tile.
func (t *Tensor) Tile(reps ...int) (*Tensor, error) {
	if slices.Contains(reps, 0) {
		return Empty(), nil
	}

	shape := t.CopyShape()

	// unify the length of shape and reps
	if len(shape) < len(reps) {
		delta := len(reps) - len(shape)
		shape = append(all(1, delta), shape...)
	} else if len(reps) < len(shape) {
		delta := len(shape) - len(reps)
		reps = append(all(1, delta), reps...)
	}

	tmpt := t.Copy()

	var r func(dim int, index []int) []float64
	r = func(dim int, index []int) []float64 {
		if len(index) == dim {
			data := []float64{}
			tmpd := []float64{}
			for i := 0; i < tmpt.Shape[dim]; i++ {
				sub, err := tmpt.SubTensor(append(index, i))
				if err != nil {
					panic(err)
				}
				tmpd = append(tmpd, sub.Data...)
			}
			data = append(data, repeat(tmpd, reps[dim])...)
			return data
		}

		data := []float64{}
		for i := 0; i < tmpt.Shape[dim]; i++ {
			tmpd := r(dim, append(index, i))
			data = append(data, tmpd...)
		}

		return data
	}

	for i := 0; i < len(shape); i++ {
		indices := tmpt.genIndices(i)
		data := []float64{}
		for _, index := range indices {
			data = append(data, r(i, index)...)
		}
		shape[i] *= reps[i]

		tt, err := Nd(data, shape...)
		if err != nil {
			return nil, err
		}

		tmpt = tt
	}

	return tmpt, nil
}

// Sum returns the sum of array elements over a given axes.
// If the empty axes is passed, calculates all values sum.
func (t *Tensor) Sum(keepdims bool, axes ...int) (*Tensor, error) {
	if len(axes) == 0 {
		// when axes is empty, sum all.
		var result float64
		for i := range t.Data {
			result += t.Data[i]
		}

		if keepdims {
			return Nd([]float64{result}, all(1, len(t.Shape))...)
		}

		return Scalar(result), nil
	}

	// check axes
	copied := make([]int, len(axes))
	copy(copied, axes)
	slices.Sort(copied)
	copied = slices.Compact(copied)
	if len(copied) != len(axes) {
		return nil, fmt.Errorf("duplicate value in axes: %v", axes)
	}

	if slices.ContainsFunc(copied, func(axis int) bool {
		return axis > len(t.Shape)-1
	}) {
		return nil, fmt.Errorf("axis out of bounds: %v", axes)
	}

	slices.Sort(axes)

	// else, sum by axis

	sumAxis := func(t *Tensor, axis int) []float64 {
		find := func(idx ...int) float64 {
			iv := t.Indices()
			for _, index := range iv {
				if slices.Equal(index.Idx, idx) {
					return index.Value
				}
			}
			panic("unexpected to come here")
		}

		indexValues := t.Indices()
		indices := make([][]int, len(indexValues))
		for i, iv := range indexValues {
			indices[i] = iv.Idx
		}
		// unique
		uindices := [][]int{}
		for _, index := range indices {
			ni := append(index[:axis], index[axis+1:]...)
			found := false
			for _, ui := range uindices {
				if slices.Equal(ui, ni) {
					found = true
				}
			}

			if !found {
				uindices = append(uindices, ni)
			}
		}

		dim := t.Shape[axis]
		result := []float64{}
		for _, index := range uindices {
			sum := 0.0
			for i := 0; i < dim; i++ {
				copied := make([]int, len(index))
				copy(copied, index)
				tmpi := append(copied[:axis], append([]int{i}, copied[axis:]...)...)
				sum += find(tmpi...)
			}
			result = append(result, sum)
		}

		return result
	}

	nt := t.Copy()

	for _, axis := range axes {
		data := sumAxis(nt, axis)
		nt.Shape[axis] = 1
		nt.Data = data
	}

	if !keepdims {
		if len(nt.Data) == 1 {
			return Scalar(nt.Data[0]), nil
		}

		return Vector(nt.Data), nil
	}

	return nt, nil
}

// Squeeze removes dimension which is 1.
func (t *Tensor) Squeeze(axes ...int) (*Tensor, error) {
	curshape := t.CopyShape()
	for _, axis := range axes {
		if curshape[axis] != 1 {
			return nil, fmt.Errorf("non-1 axis is specified")
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

	if len(newshape) == 0 {
		return Scalar(t.Data[0]), nil
	}

	if len(newshape) == 1 {
		return Vector(t.Data), nil
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
	if len(t.Shape) > len(shape) {
		return nil, fmt.Errorf("cannot broadcast: invalid shape")
	}

	newshape := make([]int, len(t.Shape))
	copy(newshape, t.Shape)
	if len(t.Shape) != len(shape) {
		delta := len(shape) - len(t.Shape)
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

func (t *Tensor) CopyShape() []int {
	ns := make([]int, len(t.Shape))
	copy(ns, t.Shape)
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
func all(n, cnt int) []int {
	r := make([]int, cnt)
	for i := 0; i < cnt; i++ {
		r[i] = n
	}
	return r
}

func repeat(data []float64, cnt int) []float64 {
	r := []float64{}
	for i := 0; i < cnt; i++ {
		r = append(r, data...)
	}
	return r
}

func toStrides(shape []int) []int {
	s := make([]int, len(shape))
	for i := range s {
		s[i] = total(shape[i+1:])
	}
	return s
}
