package tensor2

// Index returns t[indices[0], indices[1], indices[2]...].
// The returned tensor might be sharing the actual data with t,
// so modifiyig it might cause some unexpected side effect.
// func (t *Tensor) Index(indices ...int) (*Tensor, error) {
// 	// validations
// 	if t.IsScalar() {
// 		return nil, fmt.Errorf("index is not defined on scalar %v", t)
// 	}
//
// 	if len(indices) == 0 {
// 		return nil, fmt.Errorf("empty index is not allowed")
// 	}
//
// 	if len(t.Shape) < len(indices) {
// 		return nil, fmt.Errorf("too many index specified: %v", indices)
// 	}
//
// 	for i, idx := range indices {
// 		if idx < 0 || t.Shape[i]-1 < idx {
// 			return nil, fmt.Errorf("index %v is out of bounds for axis %v with size %v", idx, i, t.Shape[i])
// 		}
// 	}
//
// 	offset := t.offset
// 	for i, idx := range indices {
// 		if idx < 0 || t.Shape[i]-1 < idx {
// 			return nil, fmt.Errorf("index %v is out of bounds for axis %v with size %v", idx, i, t.Shape[i])
// 		}
//
// 		offset += t.Strides[i] * idx
// 	}
//
// 	// If len(shape) == len(indices), the single value is picked up and returned as scalar.
// 	// In this case the data is copied.
// 	if len(t.Shape) == len(indices) {
// 		return Scalar(t.data[offset]), nil
// 	}
//
// 	// Else, the shared tensor is returned.
// 	return &Tensor{data: t.data, offset: offset, Shape: t.Shape[len(indices):], Strides: t.Strides[len(indices):]}, nil
// }

//	func (t *Tensor) ListIndex(indices [][]int) (*Tensor, error) {
//		// validations
//		if t.IsScalar() {
//			return nil, fmt.Errorf("ListIndex is not defined on scalar %v", t)
//		}
//
//		if len(indices) == 0 {
//			return nil, fmt.Errorf("empty index is not allowed")
//		}
//
//		if len(t.Shape) < len(indices) {
//			return nil, fmt.Errorf("too many index specified: %v", indices)
//		}
//
//		// try broadcast each index
//
//		// check if broadcasting is possible
//		size := -1
//		for _, index := range indices {
//			if len(index) == 1 {
//				continue
//			}
//
//			if size == -1 {
//				size = len(index)
//				continue
//			}
//
//			if len(index) != size {
//				return nil, fmt.Errorf("indexing arrays could not be broadcast together with shapes: (%v), (%v)", size, len(index))
//			}
//		}
//
//		// if all index length is 1, comes here.
//		if size == -1 {
//			size = 1
//		}
//
//		// From numpy doc:
//		//     In general, the shape of the resultant array will be
//		//     the concatenation of the shape of the index array
//		//     (or the shape that all the index arrays were broadcast to) with
//		//     the shape of any unused dimensions (those not indexed) in the array being indexed.
//		// https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
//		newshape := append([]int{size}, t.Shape[len(indices):]...)
//
//		// do actual broadcast
//		for i, index := range indices {
//			if len(index) == 1 {
//				ni := make([]int, size)
//				for j := 0; j < size; j++ {
//					ni[j] = index[0]
//				}
//				indices[i] = ni
//			}
//		}
//
//		accessors := make([][]int, size)
//		for i := range size {
//			a := make([]int, len(indices))
//			for j := 0; j < len(indices); j++ {
//				a[j] = indices[j][i]
//			}
//			accessors[i] = a
//		}
//
//		data := []float64{}
//		for _, accessor := range accessors {
//			nt, err := t.Index(accessor...)
//			if err != nil {
//				return nil, err
//			}
//			data = append(data, nt.Flatten()...)
//		}
//
//		return NdShape(data, newshape...)
//	}
