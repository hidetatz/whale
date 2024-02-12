package tensor2

import "fmt"

// Slice represents start:end:step accessor to a tensor.
// Unlike Python list, negative value is not considered as the index from right but
// just converted into the default value:
//
//	start: 0
//	end:   length
//	step:  1
type Slice struct {
	Start, End, Step int
}

func (s *Slice) String() string {
	r := ""
	if 0 <= s.Start {
		r += fmt.Sprintf("%d", s.Start)
	}
	r += ":"
	if 0 <= s.End {
		r += fmt.Sprintf("%d", s.End)
	}
	r += ":"
	if 0 <= s.Step {
		r += fmt.Sprintf("%d", s.Step)
	}
	return r
}

// From creates a tensor slicing accessor like "start::".
func From(start int) *Slice { return &Slice{Start: start, End: -1, Step: 1} }

// To creates a tensor slicing accessor like ":end:".
func To(end int) *Slice { return &Slice{Start: -1, End: end, Step: 1} }

// By creates a tensor slicing accessor like "::step".
func By(step int) *Slice { return &Slice{Start: -1, End: -1, Step: step} }

// FromTo creates a tensor slicing accessor like "start:end:".
func FromTo(s, e int) *Slice { return &Slice{Start: s, End: e, Step: 1} }

// FromToBy creates a tensor slicing accessor like "start:end:step".
func FromToBy(start, end, step int) *Slice { return &Slice{Start: start, End: end, Step: step} }

// All creates a tensor slicing accessor like "::"/":".
func All() *Slice { return &Slice{Start: -1, End: -1, Step: -1} }

// Slice returns the sliced tensor of t.
// The returned tensor might be sharing the actual data with t,
// so modifiyig it might cause some unexpected side effect.
func (t *Tensor) Slice(slices ...*Slice) (*Tensor, error) {
	if t.IsScalar() {
		return nil, fmt.Errorf("slice is not defined on scalar %v", t)
	}

	if len(slices) == 0 {
		return nil, fmt.Errorf("empty slices not allowed")
	}

	if len(t.Shape) < len(slices) {
		return nil, fmt.Errorf("too many slices specified: %v", slices)
	}

	// fill slices to be the same length as t.Shape
	if len(slices) < len(t.Shape) {
		n := len(t.Shape) - len(slices)
		for i := 0; i < n; i++ {
			slices = append(slices, All())
		}
	}

	// In slicing, only the shape and strides are modified.
	// Shape is basically the same as t.Shape, but the actual data count calculated by
	// start/end/step is filled.
	// Strides is also the same as t.Strides, but it is multiplied by the step.
	newshape := make([]int, len(t.Shape))
	newstrides := make([]int, len(t.Strides))
	offset := t.offset
	for i, s := range slices {
		if s.Step == 0 {
			return nil, fmt.Errorf("slice step must not be 0: %v", slices)
		}

		// Unlike Python, negative values are not allowed.
		if s.Step < 0 {
			s.Step = 1
		}

		if s.Start < 0 {
			s.Start = 0
		}

		if s.End < 0 || t.Shape[i] < s.End {
			s.End = t.Shape[i]
		}

		if t.Shape[i] < s.Start || s.End < s.Start {
			newshape[i] = 0
		} else {
			newshape[i] = (s.End - s.Start + s.Step - 1) / s.Step
		}

		newstrides[i] = t.Strides[i] * s.Step

		if newshape[i] != 0 {
			offset += s.Start * t.Strides[i]
		}
	}

	return &Tensor{data: t.data, offset: offset, Shape: newshape, Strides: newstrides}, nil
}
