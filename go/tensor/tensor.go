package tensor

import (
	"fmt"
	"slices"
	"strings"
)

// Tensor is a multi dimensional array.
// A scalar, vector, matrix, and tensor which has more than 3-dimension can be
// represented using Tensor.
type Tensor struct {
	data   []float32
	offset int

	// Shape is a shape of the tensor.
	// Empty shape means the tensor is actually a scalar.
	// Writing Shape directly makese the tensor invalid state, so it should not be done
	// on library client side.
	Shape []int

	// Strides is a interval between the value on an axis.
	// The length of strides must be the same with Shape, and len(Shape) == len(Strides) == dimension number.
	// Writing Strides directly makes the tensor invalid state, so it should not be done
	// on library client side.
	Strides []int

	isview bool
}

// ErrResponser returns a special object error responser.
// All the tensor methods called via the returned object from ErrResponser()
// uses a method which has return type error.
// Without using ErrResponser, most methods panics on error.
func (t *Tensor) ErrResponser() *tensorErrResponser {
	return &tensorErrResponser{t: t}
}

// Equals returns if t and t2 are logically the same.
func (t *Tensor) Equals(t2 *Tensor) bool {
	return slices.Equal(t.Shape, t2.Shape) && slices.Equal(t.Ravel(), t2.Ravel())
}

// Ndim returns the dimension number.
func (t *Tensor) Ndim() int {
	return len(t.Shape)
}

// Size returns the tensor size.
func (t *Tensor) Size() int {
	return product(t.Shape)
}

// IsScalar returns true if the tensor is internally a scalar.
func (t *Tensor) IsScalar() bool {
	return len(t.Shape) == 0
}

// AsScalar returns a concrete type scalar value of the tensor.
// Note that AsScalar internally does not check if t is a scalar,
// it is caller's responsibility.
func (t *Tensor) AsScalar() float32 {
	return t.data[t.offset]
}

// IsVector returns true if the tensor is internally a vector.
func (t *Tensor) IsVector() bool {
	return len(t.Shape) == 1
}

// AsScalar returns a concrete type vector slice of the tensor.
// Note that AsVector internally does not check if t is a vector,
// it is caller's responsibility.
func (t *Tensor) AsVector() []float32 {
	indices := cartesianIdx(t.Shape)
	result := make([]float32, t.Shape[0])
	for i, index := range indices {
		f := t.Index(index...)
		result[i] = f.AsScalar()
	}
	return result
}

// IsMatrix returns true if the tensor is internally a matrix.
func (t *Tensor) IsMatrix() bool {
	return len(t.Shape) == 2
}

// Copy returns a copy of t.
func (t *Tensor) Copy() *Tensor {
	ndata := make([]float32, len(t.data))
	copy(ndata, t.data)

	nshape := make([]int, len(t.Shape))
	copy(nshape, t.Shape)

	nstrides := make([]int, len(t.Strides))
	copy(nstrides, t.Strides)

	return &Tensor{data: ndata, offset: t.offset, Shape: nshape, Strides: nstrides, isview: false}
}

// Ravel returns a flattened slice. The returned slice might be sharing
// the same memory with t. If you want to modify returned slice, you'd better
// use Flatten instead.
func (t *Tensor) Ravel() []float32 {
	arr, _ := t.ravel()
	return arr
}

// Flatten returns flattend, copied 1-D array.
func (t *Tensor) Flatten() []float32 {
	arr, view := t.ravel()
	if view {
		return copySlice(arr)
	}

	return arr
}

func (t *Tensor) ravel() (result []float32, view bool) {
	switch {
	case t.IsScalar():
		return []float32{t.AsScalar()}, false

	case t.IsVector():
		// fast path, lucky
		if len(t.data) == t.Shape[0] {
			return t.data, true
		}
	case t.IsMatrix():
		// fast path, lucky
		if len(t.data) == product(t.Shape) && t.Strides[0] == t.Shape[1] && t.Strides[1] == 1 {
			return t.data, true
		}
	}

	indices := cartesian(t.Shape)
	result = make([]float32, len(indices))
	for i, index := range indices {
		rawIdx := t.offset
		for j, idx := range index {
			rawIdx += t.Strides[j] * idx
		}
		result[i] = t.data[rawIdx]
	}

	return result, false
}

// Raw prints the tensor raw internal structure for debug purpose.
func (t *Tensor) Raw() string {
	return fmt.Sprintf("{data: %v, offset: %v, Shape: %v, Strides: %v, isview: %v}", t.data, t.offset, t.Shape, t.Strides, t.isview)
}

// String implements fmt.Stringer interface.
func (t *Tensor) String() string {
	return t.tostring(true)
}

// OnelineString prints the tensor in one line string.
func (t *Tensor) OnelineString() string {
	return t.tostring(false)
}

func (t *Tensor) tostring(linebreak bool) string {
	if t.IsScalar() {
		return fmt.Sprintf("%v", t.AsScalar())
	}

	if slices.Contains(t.Shape, 0) {
		if linebreak {
			return fmt.Sprintf("([], shape=%v)", t.Shape)
		} else {
			return "[]"
		}
	}

	if t.IsVector() {
		return fmt.Sprintf("%v", strings.Join(strings.Fields(fmt.Sprint(t.AsVector())), ", "))
	}

	tostr := func(fs []float32) []string {
		ss := make([]string, len(fs))
		for i, f := range fs {
			ss[i] = fmt.Sprintf("%v", f)
		}
		return ss
	}

	var w func(index []int) string
	w = func(index []int) string {
		indent := strings.Repeat("  ", len(index))

		if len(index) == len(t.Shape)-1 {
			data := t.Index(intsToIndices(index)...).Ravel()
			vals := strings.Join(tostr(data), ", ")
			if linebreak {
				return fmt.Sprintf("%s[%v]", indent, vals)
			} else {
				return fmt.Sprintf("[%v]", vals)
			}
		}

		outer := make([]string, t.Shape[len(index)])
		for i := range t.Shape[len(index)] {
			inner := w(append(index, i))
			outer[i] = inner
		}

		if linebreak {
			return fmt.Sprintf("%s[\n", indent) + strings.Join(outer, "\n") + fmt.Sprintf("\n%s]", indent)
		} else {
			return "[" + strings.Join(outer, ", ") + "]"
		}
	}

	if linebreak {
		return w([]int{}) + "\n"
	} else {
		return w([]int{})
	}
}
