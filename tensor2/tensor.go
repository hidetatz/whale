package tensor2

import (
	"fmt"
	"slices"
	"strings"
)

// Tensor is a multi dimensional array.
// A scalar, vector, matrix, and tensor which has more than 3-dimension can be
// represented using Tensor.
type Tensor struct {
	data    []float64
	offset  int

	// Shape is a shape of the tensor.
	// Empty shape means the tensor is actually a scalar.
	// Writing Shape directly makese the tensor invalid state, so it should not be done
	// on library client side.
	Shape   []int

	// Strides is a interval between the value on an axis.
	// The length of strides must be the same with Shape, and len(Shape) == len(Strides) == dimension number.
	// Writing Strides directly makes the tensor invalid state, so it should not be done
	// on library client side.
	Strides []int

	isview  bool

	// RespErr is a special object for those who want this package to use methods which
	// has return error type if it happens.
	// Without using RespErr, most methods panics on error.
	RespErr *errResponser
}

// Equals returns if t and t2 are logically the same.
func (t *Tensor) Equals(t2 *Tensor) bool {
	return slices.Equal(t.Shape, t2.Shape) && slices.Equal(t.Flatten(), t2.Flatten())
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
func (t *Tensor) AsScalar() float64 {
	return t.data[t.offset]
}

// IsVector returns true if the tensor is internally a vector.
func (t *Tensor) IsVector() bool {
	return len(t.Shape) == 1
}

// AsScalar returns a concrete type vector slice of the tensor.
// Note that AsVector internally does not check if t is a vector,
// it is caller's responsibility.
func (t *Tensor) AsVector() []float64 {
	indices := cartesianIdx(t.Shape)
	result := make([]float64, t.Shape[0])
	for i, index := range indices {
		f := MustGet(t.Index(index...))
		result[i] = f.AsScalar()
	}
	return result
}

// Copy returns a copy of t.
func (t *Tensor) Copy() *Tensor {
	ndata := make([]float64, len(t.data))
	copy(ndata, t.data)

	nshape := make([]int, len(t.Shape))
	copy(nshape, t.Shape)

	nstrides := make([]int, len(t.Strides))
	copy(nstrides, t.Strides)

	return &Tensor{data: ndata, offset: t.offset, Shape: nshape, Strides: nstrides}
}

// Flatten returns flattend 1-D array.
func (t *Tensor) Flatten() []float64 {
	if t.IsScalar() {
		return []float64{t.AsScalar()}
	}

	// fast path: no need to calculate cartesian from strides
	if !t.isview {
		return copySlice(t.data)
	}

	indices := cartesian(t.Shape)
	result := make([]float64, len(indices))
	for i, index := range indices {
		rawIdx := t.offset
		for j, idx := range index {
			rawIdx += t.Strides[j] * idx
		}
		result[i] = t.data[rawIdx]
	}
	return result
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

	tostr := func(fs []float64) []string {
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
			data := MustGet(t.Index(intsToIndices(index)...)).Flatten()
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

