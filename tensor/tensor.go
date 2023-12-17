package tensor

import (
	"fmt"
	"slices"
	"strings"
)

type Tensor struct {
	Data    []float64
	shape   []int
	strides []int
}

func FromScalar(s float64) *Tensor {
	return &Tensor{Data: []float64{s}}
}

func FromVector(v []float64, shape int) (*Tensor, error) {
	if len(v) != shape {
		return nil, fmt.Errorf("shape mismatch with vector length")
	}
	return &Tensor{Data: v, shape: []int{shape}, strides: []int{1}}, nil
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

	t := &Tensor{Data: data, shape: shape}

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

func Zeros(shape ...int) *Tensor {
	data := make([]float64, total(shape)) // initialized by 0
	t, _ := Nd(data, shape...)            // error never happens
	return t
}

func Ones(shape ...int) *Tensor {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = 1
	}
	t, _ := Nd(data, shape...) // error never happens
	return t
}

func All(v float64, shape ...int) *Tensor {
	data := make([]float64, total(shape))
	for i := range data {
		data[i] = v
	}
	t, _ := Nd(data, shape...) // error never happens
	return t
}

func ArangeTo(to int) *Tensor {
	var data []float64
	if to < 0 {
		data = seq(to, 0)
		slices.Reverse(data)
	} else {
		data = seq(0, to)
	}

	t, _ := FromVector(data, len(data)) // error never happens
	return t
}

func ArangeFrom(from, to int) *Tensor {
	var data []float64
	if to < from {
		data = seq(to, from)
		slices.Reverse(data)
	} else {
		data = seq(from, to)
	}

	t, _ := FromVector(data, len(data)) // error never happens
	return t
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

func (t *Tensor) IsScalar() bool {
	return len(t.shape) == 0
}

func (t *Tensor) IsVector() bool {
	return len(t.shape) == 1
}

func (t *Tensor) RawString() string {
	return fmt.Sprintf("Tensor{Data: %v, shape: %v, strides: %v}", t.Data, t.shape, t.strides)
}

func (t *Tensor) String() string {
	if t.IsScalar() {
		return fmt.Sprintf("%v (scalar)", t.Data[0])
	}

	if t.IsVector() {
		return fmt.Sprintf("%v (vector)", t.Data)
	}

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
				sb.WriteString(fmt.Sprintf("%v", t.Data[idx]))
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

		sb.WriteString(fmt.Sprintf("%s]", indent))
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

	if !slices.Equal(t.Data, t2.Data) {
		return false
	}

	return true
}

func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	t2 := t.Copy()
	if total(t2.shape) != total(shape) {
		return nil, fmt.Errorf("cannot reshape: the data size mismatch")
	}

	return Nd(t2.Data, shape...)
}

func (t *Tensor) Copy() *Tensor {
	ndata := make([]float64, len(t.Data))
	copy(ndata, t.Data)

	nshape := make([]int, len(t.shape))
	copy(nshape, t.shape)

	nstrides := make([]int, len(t.strides))
	copy(nstrides, t.strides)

	return &Tensor{
		Data:    ndata,
		shape:   nshape,
		strides: nstrides,
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

func (t *Tensor) Iterator(axis int) (*Iterator, error) {
	if axis > len(t.strides) {
		return nil, fmt.Errorf("axis mismatch")
	}

	return &Iterator{t: t, axis: axis}, nil
}

func (t *Tensor) Repeat(times, axis int) (*Tensor, error) {
	newshape := t.CopyShape()
	newshape[axis] *= times

	newdata := []float64{}

	iter, err := t.Iterator(axis)
	if err != nil {
		return nil, err
	}

	for iter.HasNext() {
		data := iter.Next()
		for i := 0; i < times; i++ {
			newdata = append(newdata, data...)
		}
	}

	return Nd(newdata, newshape...)
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
			for i := 0; i < len(t.Shape()); i++ {
				shape = append(shape, 1)
			}

			return Nd([]float64{result}, shape...)
		}

		return FromScalar(result), nil
	}

	// else, sum by axis
	curshape := t.CopyShape()

	slices.Sort(axes)
	slices.Reverse(axes) // ordered desc
	nt := t.Copy()
	for _, axis := range axes {
		axisdim := curshape[axis]

		datalen := total(nt.shape) / axisdim
		newdata := make([]float64, datalen)

		stride := nt.strides[axis]
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
			return nil, fmt.Errorf("Squeeze: axis which is not 1 is specified")
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

func (t *Tensor) CopyShape() []int {
	ns := make([]int, len(t.shape))
	copy(ns, t.shape)
	return ns
}

func (t *Tensor) CopyStrides() []int {
	ns := make([]int, len(t.strides))
	copy(ns, t.strides)
	return ns
}

func seq(from, to int) []float64 {
	r := make([]float64, to-from)
	for i := from; i < to; i++ {
		r[i-from] = float64(i)
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