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

	indices := cartesian(t.Shape)
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

	data := t.Flatten()
	var sb strings.Builder

	var w func(index []int)
	w = func(index []int) {
		indent := strings.Repeat("  ", len(index))

		if len(index) == len(t.Shape)-1 {
			vars := []string{}
			for i := 0; i < t.Shape[len(t.Shape)-1]; i++ {
				idx := 0
				for j := range index {
					idx += index[j] * t.Strides[j]
				}
				idx += i * t.Strides[len(t.Strides)-1]
				vars = append(vars, fmt.Sprintf("%.2f", data[idx]))
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