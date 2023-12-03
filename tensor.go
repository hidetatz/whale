package whale

import (
	"fmt"
	"strings"
)

type node struct {
	values   []float64
	children []*node
}

func (n *node) isleaf() bool {
	return len(n.values) > 0
}

type Tensor struct {
	root  *node
	shape []int
	dim   int
}

func FromScalar(data float64) *Tensor {
	root := &node{values: []float64{data}}
	return &Tensor{root: root}
}

func FromVector(data []float64, shape int) *Tensor {
	root := &node{values: data}
	return &Tensor{root: root, shape: []int{shape}}
}

func Nd(data []float64, shape ...int) (*Tensor, error) {
	// scalar
	if len(shape) == 0 {
		return FromScalar(data[0]), nil
	}

	// vector
	if len(shape) == 1 {
		if len(data) != shape[0] {
			return nil, fmt.Errorf("elements count and shape mismatch")
		}
		return FromVector(data, shape[0]), nil
	}

	// matrix/tensor
	elements := total(shape)
	if len(data) != elements {
		return nil, fmt.Errorf("elements count and shape mismatch")
	}

	t := &Tensor{shape: shape, root: &node{}, dim: len(shape)}
	last := []*node{t.root}

	for i, dim := range shape {
		// The last. Set actual values
		if i == len(shape)-1 {
			for j, l := range last {
				l.values = data[dim*j : (j+1)*dim]
			}
			break
		}

		newlast := []*node{}
		for _, l := range last {
			children := []*node{}
			for k := 0; k < dim; k++ {
				children = append(children, &node{})
			}
			l.children = children
			newlast = append(newlast, children...)
		}

		last = newlast

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
	return TensorNd(data, shape...)
}

func (t *Tensor) Shape() []int {
	return t.shape
}

func (t *Tensor) Dim() int {
	return t.dim
}

func (t *Tensor) String() string {
	s := ""
	var printNode func(n *node, depth int)
	printNode = func(n *node, depth int) {
		if n.isleaf() {
			s += fmt.Sprintf("%s%v\n", strings.Repeat("  ", depth), n.values)
			return
		}

		s += fmt.Sprintf("%s[\n", strings.Repeat("  ", depth))
		for _, c := range n.children {
			printNode(c, depth+1)
		}
		s += fmt.Sprintf("%s]\n", strings.Repeat("  ", depth))

	}
	printNode(t.root, 0)
	return s
}

func (t *Tensor) data() []float64 {
	data := make([]float64, total(t.shape))
	appenddata := func(n *node)
	appenddata = func(n *node) {
		if n.isleaf() {
			for _, v := range n.values {
				data = append(data, v...)
			}
			return
		}
		for _, c := range n.children {
			appenddata(c)
		}
	}
	appenddata(t.root)
	return data
}

func (t *Tensor) Equals(t2 *Tensor) bool {
	if !t.SameShape(t2) {
		return false
	}

	// just in case
	if t.dim != t2.dim {
		return false
	}

	if len(t.data) != len(t2.data) {
		return false
	}

	for i := range t.data {
		if t.data[i] != t2.data[i] {
			return false
		}
	}

	return true
}

func (t *Tensor) SameShape(t2 *Tensor) bool {
	if len(t.shape) != len(t2.shape) {
		return false
	}

	for i := range t.shape {
		if t.shape[i] != t2.shape[i] {
			return false
		}
	}

	return true
}

func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if total(t.shape) != total(shape) {
		return nil, fmt.Errorf("cannot reshape: the data size mismatch")
	}

	data := t.data()
	return Nd(data, shape), nil
}

func (t *Tensor) Transpose() *Tensor {
	// scalar
	if t.dim == 0 {
		return FromScalar(t.data()[0])
	}

	// vector
	if t.dim == 1 {
		d = t.data()
		return FromVector(d, len(d))
	}

}

func (t *Tensor) BroadcastTo(shape ...int) (*Tensor, error) {
	if total(t.shape) > total(shape) {
		return nil, fmt.Errorf("cannot broadcast: ndim must be smaller then the given one")
	}

	newshape := make([]int, len(t.shape))
	copy(newshape, t.shape)
	if len(t.shape) != len(shape) {
		delta := len(shape) - len(newshape)
		for i := 0; i < delta; i++ {
			// push 1 to the head until the dim gets the same
			newshape = append([]int{1}, newshape...)
		}
	}

	nt, err := t.Reshape(newshape...)
	if err != nil {
		return nil, err
	}
	fmt.Println(nt, nt.data, nt.shape, nt.dim, shape)
	nt, err = nt.Tile(shape...)
	if err != nil {
		return nil, err
	}

	return nt, nil
}


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
