package tensor2

import (
	"fmt"
	"slices"
	"strings"
)

type Tensor struct {
	data    []float64
	offset  int
	Shape   []int
	Strides []int
}

// Equals returns if t and t2 are the same.
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

func (t *Tensor) AsScalar() float64 {
	return t.data[t.offset]
}

// IsVector returns true if the tensor is internally a vector.
func (t *Tensor) IsVector() bool {
	return len(t.Shape) == 1
}

func (t *Tensor) AsVector() []float64 {
	indices := cartesianIdx(t.Shape)
	result := make([]float64, t.Shape[0])
	for i, index := range indices {
		f := Must(t.Index(index...))
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

	indices := cartesianIdx(t.Shape)
	result := make([]float64, len(indices))
	for i, index := range indices {
		s, err := t.Index(index...)
		if err != nil {
			panic(fmt.Errorf("Flatten: index access problem: %v", index))
		}

		if !s.IsScalar() {
			panic("Flatten: non scalar")
		}

		result[i] = s.AsScalar()
	}
	return result
}

// String implements Stringer interface.
func (t *Tensor) String() string {
	if t.IsScalar() {
		return fmt.Sprintf("%v", t.AsScalar())
	}

	if slices.Contains(t.Shape, 0) {
		return fmt.Sprintf("([], shape=%v)", t.Shape)
	}

	if t.IsVector() {
		return fmt.Sprintf("%v", strings.Join(strings.Fields(fmt.Sprint(t.AsVector())), ", "))
	}

	var sb strings.Builder
	var w func(index []int)
	w = func(index []int) {
		indent := strings.Repeat("  ", len(index))

		// at last, print actual values
		if len(index) == len(t.Shape)-1 {
			data := Must(t.Index(intsToIndices(index)...)).Flatten()
			sb.WriteString(
				fmt.Sprintf("%s%v\n", indent, strings.Join(strings.Fields(fmt.Sprint(data)), ", ")),
			)
			return
		}

		// else, print "[",  internal data, and "]" recursively
		sb.WriteString(fmt.Sprintf("%s[\n", indent))
		for i := range t.Shape[len(index)] {
			w(append(index, i))
		}
		sb.WriteString(fmt.Sprintf("%s]\n", indent))
	}

	w([]int{})
	return sb.String()
}

func (t *Tensor) Raw() string {
	return fmt.Sprintf("{data: %v, offset: %v, Shape: %v, Strides: %v}", t.data, t.offset, t.Shape, t.Strides)
}
