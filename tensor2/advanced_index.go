package tensor2

func (t *Tensor) advancedIndex(args ...*IndexArg) (*Tensor, error) {
	panic("unimplemented")
	// // try broadcast each index
	// broadcasted, err := tryBroadcast()
	//
	// // check if broadcasting is possible
	// size := -1
	//
	//	for _, index := range indices {
	//		if len(index) == 1 {
	//			continue
	//		}
	//
	//		if size == -1 {
	//			size = len(index)
	//			continue
	//		}
	//
	//		if len(index) != size {
	//			return nil, fmt.Errorf("indexing arrays could not be broadcast together with shapes: (%v), (%v)", size, len(index))
	//		}
	//	}
	//
	// // if all index length is 1, comes here.
	//
	//	if size == -1 {
	//		size = 1
	//	}
	//
	// // From numpy doc:
	// //     In general, the shape of the resultant array will be
	// //     the concatenation of the shape of the index array
	// //     (or the shape that all the index arrays were broadcast to) with
	// //     the shape of any unused dimensions (those not indexed) in the array being indexed.
	// // https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
	// newshape := append([]int{size}, t.Shape[len(indices):]...)
	//
	// // do actual broadcast
	//
	//	for i, index := range indices {
	//		if len(index) == 1 {
	//			ni := make([]int, size)
	//			for j := 0; j < size; j++ {
	//				ni[j] = index[0]
	//			}
	//			indices[i] = ni
	//		}
	//	}
	//
	// accessors := make([][]int, size)
	//
	//	for i := range size {
	//		a := make([]int, len(indices))
	//		for j := 0; j < len(indices); j++ {
	//			a[j] = indices[j][i]
	//		}
	//		accessors[i] = a
	//	}
	//
	// data := []float64{}
	//
	//	for _, accessor := range accessors {
	//		nt, err := t.Index(accessor...)
	//		if err != nil {
	//			return nil, err
	//		}
	//		data = append(data, nt.Flatten()...)
	//	}
	//
	// return NdShape(data, newshape...)
}
