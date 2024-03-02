package tensor2

// type Ufunc interface {
// 	At(a *Tensor, indices []*Tensor, b *Tensor) error
// }
//
// type add struct{}
//
// func (add *add) At(a *Tensor, indices []*Tensor, b *Tensor) error {
// 	is, err := c.Indices(indices...)
// 	if err != nil {
// 		return nil, err
// 	}
//
// 	broadcasted, err := val.Copy().BroadcastTo(is.CopyShape()...)
// 	if err != nil {
// 		return nil, err
// 	}
//
// 	intIndices := [][]int{}
// 	for i := 0; i < len(indices[0].Data); i++ {
// 		ints := make([]int, len(indices))
// 		for j, index := range indices {
// 			ints[j] = int(index.Data[i])
// 		}
//
// 		intIndices = append(intIndices, ints)
// 	}
//
// 	startswith := func(a, b []int) bool {
// 		for i := range a {
// 			if a[i] != b[i] {
// 				return false
// 			}
// 		}
// 		return true
// 	}
//
// 	vis := c.ValueIndices()
// 	for i, ints := range intIndices {
// 		for _, vi := range vis {
// 			// todo: can be optimized?
// 			if startswith(ints, vi.Idx) {
// 				vi.Value += broadcasted.Data[i]
// 			}
// 		}
// 	}
//
// 	data := make([]float64, len(vis))
// 	for i, vi := range vis {
// 		data[i] = vi.Value
// 	}
//
// 	c.Data = data
//
// 	return c, nil
// }
