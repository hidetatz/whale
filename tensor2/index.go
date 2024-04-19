package tensor2

import (
	"fmt"
	"slices"
)

type indexResult struct {
	t           *Tensor
	origIndices []int
}

func (r *indexResult) String() string {
	return fmt.Sprintf(`{
  t: %v,
  origIndices: %v
}`, r.t, r.origIndices)
}

func (t *Tensor) Index(args ...*IndexArg) (*Tensor, error) {
	r, err := t.index(args...)
	if err != nil {
		return nil, err
	}
	return r.t, nil
}

func (t *Tensor) index(args ...*IndexArg) (*indexResult, error) {
	if t.IsScalar() {
		return nil, fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(args) == 0 {
		return nil, fmt.Errorf("index accessor must not be empty")
	}

	if t.Ndim() < len(args) {
		return nil, fmt.Errorf("too many index accessors specified: %v", args)
	}

	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })

	if advanced {
		return t.advancedIndex(args...)
	}

	return t.basicIndex(args...)
}

func (t *Tensor) basicIndex(args ...*IndexArg) (*indexResult, error) {
	// fill args to be the same length as ndim
	if len(args) < t.Ndim() {
		for _ = range t.Ndim() - len(args) {
			args = append(args, All())
		}
	}

	newshape := make([]int, len(t.Shape))
	newstrides := make([]int, len(t.Strides))
	offset := t.offset
	for i, arg := range args {
		if arg.typ == _int {
			if t.Shape[i]-1 < arg.i {
				return nil, fmt.Errorf("index out of bounds for axis %v with size %v", i, t.Shape[i])
			}

			offset += arg.i * t.Strides[i]
			newshape[i] = -1   // dummy value
			newstrides[i] = -1 // dummy value
			continue
		}

		// coming here means the arg type is slice

		if err := arg.s.tidy(t.Shape[i]); err != nil {
			return nil, err
		}

		if t.Shape[i] < arg.s.start || arg.s.end < arg.s.start {
			newshape[i] = 0
		} else {
			newshape[i] = arg.s.size()
		}

		newstrides[i] = t.Strides[i] * arg.s.step

		if newshape[i] != 0 {
			offset += arg.s.start * t.Strides[i]
		}
	}

	deldummy := func(n int) bool { return n == -1 }
	newshape = slices.DeleteFunc(newshape, deldummy)
	newstrides = slices.DeleteFunc(newstrides, deldummy)

	newtensor := &Tensor{data: t.data, offset: offset, Shape: newshape, Strides: newstrides}

	var origIndices []int
	if newtensor.IsScalar() {
		origIndices = []int{offset}
	} else {
		indices := cartesian(newshape)
		origIndices = make([]int, len(indices))
		for i, index := range indices {
			origIndices[i] = offset
			for j, idx := range index {
				origIndices[i] += idx * newstrides[j]
			}
		}
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}

func (t *Tensor) advancedIndex(args ...*IndexArg) (*indexResult, error) {
	containslice := slices.ContainsFunc(args, func(a *IndexArg) bool { return a.typ == _slice })
	if containslice {
		return t.advancedAndBasicCombinedIndex(args...)
	}

	/*
	 * First, determine new shape.
	 */

	// pick up tensors in args
	ts := []*Tensor{}
	for _, arg := range args {
		if arg.typ == _tensor {
			ts = append(ts, arg.t)
		}
	}

	if len(ts) == 0 {
		// must not come here
		panic("no tensor specified in advanced index")
	}

	// determine the broadcasted shape
	broadcastedshape, err := CanBroadcast(ts)
	if err != nil {
		return nil, err
	}

	// if there's no slice in args, the shape follows advanced indexing rule:
	//     In general, the shape of the resultant array will be the concatenation of the shape of
	//     the index array (or the shape that all the index arrays were broadcast to)
	//     with the shape of any unused dimensions (those not indexed) in the array being indexed.
	//     (https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
	newshape := append(broadcastedshape, t.Shape[len(args):]...)

	/*
	 * pick up values.
	 */

	flatargs := make([][]int, len(args))

	// first, broadcast args
	for i, arg := range args {
		te := arg.t
		if arg.typ == _int {
			te = Scalar(float64(arg.i))
		}

		b, err := te.BroadcastTo(broadcastedshape...)
		if err != nil {
			// must not come here
			panic(err)
		}
		flatargs[i] = toint(b.Flatten())
	}

	// pick up elements on each dimension
	indices := make([][]int, product(broadcastedshape))
	for i := range indices {
		index := make([]int, len(args))
		for j, arg := range flatargs {
			index[j] = arg[i]
		}
		indices[i] = index
	}

	// do Index and copy values
	var origIndices []int
	var r []float64
	for _, idx := range indices {
		t2, err := t.Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)

		origIdx := 0
		for i, id := range idx {
			origIdx += id * t.Strides[i]
		}
		origIndices = append(origIndices, origIdx)
	}

	newtensor, err := NdShape(r, newshape...)
	if err != nil {
		return nil, err
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}

func (t *Tensor) advancedAndBasicCombinedIndex(args ...*IndexArg) (*indexResult, error) {
	/*
	 * First, determine new shape.
	 */

	// pick up tensors in args
	ts := []*Tensor{}
	for i, arg := range args {
		if arg.typ == _tensor {
			ts = append(ts, arg.t)
			continue
		}

		if arg.typ == _slice {
			arg.s.tidy(t.Shape[i])
		}
	}

	if len(ts) == 0 {
		// must not come here
		panic("no tensor specified in advanced index")
	}

	// determine the broadcasted shape
	broadcastedshape, err := CanBroadcast(ts)
	if err != nil {
		return nil, err
	}

	// make sure if tensor/int argument is "separated" by slice.
	// https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
	isseparated := func(args []*IndexArg) bool {
		if len(args) < 3 {
			return false
		}

		for i := 1; i < len(args)-1; i++ {
			// if arg is not a slice, it is still not separated
			if args[i].typ != _slice {
				continue
			}

			// arg is a slice, but left is also a slice. still not separated
			if args[i-1].typ == _slice {
				continue
			}

			// arg is a slice and left is not a slice. look for rest and find tensor/int.
			// if found, it is separated
			for j := i + 1; j < len(args); j++ {
				if args[j].typ == _tensor || args[j].typ == _int {
					return true
				}
			}

			return false
		}

		return false
	}

	var newshape []int

	separated := isseparated(args)
	if separated {
		// according to the above document:
		//     In the first case, the dimensions resulting from the advanced indexing operation
		//     come first in the result array, and the subspace dimensions after that.
		newshape = broadcastedshape
		for _, arg := range args {
			if arg.typ != _slice {
				continue
			}
			newshape = append(newshape, arg.s.size())
		}
		newshape = append(newshape, t.Shape[len(args):]...)
	} else {
		// according to the above document:
		//     In the second case, the dimensions from the advanced indexing operations are
		//     inserted into the result array at the same spot as they were in the initial array
		//     (the latter logic is what makes simple advanced indexing behave just like slicing).

		appended := false
		for _, arg := range args {
			if arg.typ == _slice {
				newshape = append(newshape, arg.s.size())
				continue
			}

			if !appended {
				newshape = slices.Concat(newshape, broadcastedshape)
				appended = true
			}

		}
		newshape = slices.Concat(newshape, t.Shape[len(args):])
	}

	/*
	 * pick up values.
	 */

	var indices [][]int
	var current []int

	var f func(pos int, follow bool, followidx int)
	f = func(pos int, follow bool, followidx int) {
		if pos == len(args) {
			indices = append(indices, copySlice(current))
			return
		}

		arg := args[pos]
		switch arg.typ {
		case _int:
			if follow {
				current = append(current, arg.i)
				f(pos+1, follow, followidx)
				current = current[:len(current)-1]
			} else {
				for i := range product(broadcastedshape) {
					current = append(current, arg.i)
					f(pos+1, true, i)
					current = current[:len(current)-1]
				}
			}
		case _slice:
			sindices := arg.s.indices()
			for _, sidx := range sindices {
				current = append(current, sidx)
				f(pos+1, follow, followidx)
				current = current[:len(current)-1]
			}
		case _tensor:
			tindices := Must(arg.t.BroadcastTo(broadcastedshape...)).Flatten()
			if follow {
				current = append(current, int(tindices[followidx]))
				f(pos+1, follow, followidx)
				current = current[:len(current)-1]
			} else {
				for i, tidx := range tindices {
					current = append(current, int(tidx))
					f(pos+1, true, i)
					current = current[:len(current)-1]
				}
			}
		}
	}
	f(0, false, 0)

	var r []float64
	var origIndices []int
	for _, idx := range indices {
		t2, err := t.Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)

		origIdx := 0
		for i, id := range idx {
			origIdx += id * t.Strides[i]
		}
		origIndices = append(origIndices, origIdx)
	}

	newtensor, err := NdShape(r, newshape...)
	if err != nil {
		return nil, err
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}
