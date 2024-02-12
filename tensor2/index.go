package tensor2

import "fmt"

// Index returns t[indices[0], indices[1], indices[2]...].
// The returned tensor might be sharing the actual data with t,
// so modifiyig it might cause some unexpected side effect.
func (t *Tensor) Index(indices ...int) (*Tensor, error) {
	// validations
	if t.IsScalar() {
		return nil, fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(indices) == 0 {
		return nil, fmt.Errorf("empty index is not allowed")
	}

	if len(t.Shape) < len(indices) {
		return nil, fmt.Errorf("too many index specified: %v", indices)
	}

	for i, idx := range indices {
		if idx < 0 || t.Shape[i]-1 < idx {
			return nil, fmt.Errorf("index %v is out of bounds for axis %v with size %v", idx, i, t.Shape[i])
		}
	}

	offset := t.offset
	for i, idx := range indices {
		if idx < 0 || t.Shape[i]-1 < idx {
			return nil, fmt.Errorf("index %v is out of bounds for axis %v with size %v", idx, i, t.Shape[i])
		}

		offset += t.Strides[i] * idx
	}

	// If len(shape) == len(indices), the single value is picked up and returned as scalar.
	// In this case the data is copied.
	if len(t.Shape) == len(indices) {
		return Scalar(t.data[offset]), nil
	}

	// Else, the shared tensor is returned.
	return &Tensor{data: t.data, offset: offset, Shape: t.Shape[len(indices):], Strides: t.Strides[len(indices):]}, nil
}
