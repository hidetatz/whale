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
	return fmt.Sprintf("t: %v, origIndices: %v", r.t.OnelineString(), r.origIndices)
}

func (t *Tensor) Index(args ...*IndexArg) *Tensor {
	return MustGet(t.ErrResponser().Index(args...))
}

func (er *tensorErrResponser) Index(args ...*IndexArg) (*Tensor, error) {
	return er.t.indexForRead(args...)
}

func (t *Tensor) checkIndexArgs(args ...*IndexArg) error {
	if t.IsScalar() {
		return fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(args) == 0 {
		return fmt.Errorf("index accessor must not be empty")
	}

	if t.Ndim() < len(args) {
		return fmt.Errorf("too many index accessors specified: %v", args)
	}

	return nil
}

func (t *Tensor) indexForRead(args ...*IndexArg) (*Tensor, error) {
	if err := t.checkIndexArgs(args...); err != nil {
		return nil, err
	}

	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })
	if advanced {
		r, err := t.advancedIndex(args...)
		if err != nil {
			return nil, err
		}

		return r.t, nil
	}

	return t.basicIndexForRead(args...)
}

func (t *Tensor) indexForWrite(args ...*IndexArg) (*indexResult, error) {
	if err := t.checkIndexArgs(args...); err != nil {
		return nil, err
	}

	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })
	if advanced {
		return t.advancedIndex(args...)
	}

	return t.basicIndexForWrite(args...)
}

// basicIndexForRead is a indexing method to work the same as numpy's basic indexing.
// https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing
// Basic indexing happens when index arguments are only consists of integer and slice.
// In basic indexing, the returned tensor might be a view of the original tensor t (might be sharing internal data with t).
// This means only reading returned tensor is safe, but modifying it can break original t.
func (t *Tensor) basicIndexForRead(args ...*IndexArg) (*Tensor, error) {
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

	return t.view(offset, newshape, newstrides), nil
}

func (t *Tensor) basicIndexForWrite(args ...*IndexArg) (*indexResult, error) {
	newtensor, err := t.basicIndexForRead(args...)
	if err != nil {
		return nil, err
	}

	var origIndices []int
	if newtensor.IsScalar() {
		origIndices = []int{newtensor.offset}
	} else {
		indices := cartesian(newtensor.Shape)
		origIndices = make([]int, len(indices))
		for i, index := range indices {
			origIndices[i] = newtensor.offset
			for j, idx := range index {
				origIndices[i] += idx * newtensor.Strides[j]
			}
		}
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}

// advancedIndex is a indexing method which works the same as numpy's advanced indexing.
// https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
// This indexing happens when indexing arguments contains at least one tensor.
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
		t2, err := t.ErrResponser().Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)

		combi := [][]int{}
		for i := 0; i < len(idx); i++ {
			combi = append(combi, []int{idx[i]})
		}
		if len(idx) < t.Ndim() {
			for i := len(idx); i < t.Ndim(); i++ {
				combi = append(combi, until(t.Shape[i]))
			}
		}

		rawindices := cartesians(combi)
		for _, rawidx := range rawindices {
			origIdx := t.offset
			for i := range rawidx {
				origIdx += rawidx[i] * t.Strides[i]
			}
			origIndices = append(origIndices, origIdx)
		}
	}

	newtensor, err := RespErr.NdShape(r, newshape...)
	if err != nil {
		return nil, err
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}

// advancedAndBasicCombinedIndex is a special indexing method which is a part of advanced indexing,
// but in particular arguments contains both slice and tensor.
// https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
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

	type argpair struct {
		idx  int
		vals []int
	}

	reorderedArgpairs := [][]*argpair{}
	idx := -1

	if separated {
		reorderedArgpairs = append(reorderedArgpairs, []*argpair{})
		idx = 0
	}

	for i, arg := range args {
		switch arg.typ {
		case _int, _tensor:
			if idx == -1 {
				reorderedArgpairs = append(reorderedArgpairs, []*argpair{})
				idx = i
			}

			vals := all(arg.i, product(broadcastedshape))
			if arg.typ == _tensor {
				tb := MustGet(arg.t.BroadcastTo(broadcastedshape...))
				vals = toint(tb.Flatten())
			}
			reorderedArgpairs[idx] = append(
				reorderedArgpairs[idx],
				&argpair{idx: i, vals: vals},
			)

		case _slice:
			reorderedArgpairs = append(reorderedArgpairs, []*argpair{&argpair{idx: i, vals: arg.s.indices()}})
		}
	}

	var indices [][]int
	var f func(pos int, cur []int)
	f = func(pos int, cur []int) {
		if len(reorderedArgpairs) == pos {
			indices = append(indices, copySlice(cur))
			return
		}

		argpairs := reorderedArgpairs[pos]

		if pos == 0 {
			cur = make([]int, len(args))
		}

		if len(args) > 1 {
			for i := range len(argpairs[0].vals) {
				for _, ap := range argpairs {
					cur[ap.idx] = ap.vals[i]
				}
				f(pos+1, cur)
			}
		} else {
			ap := argpairs[0]
			for _, v := range ap.vals {
				cur[ap.idx] = v
				f(pos+1, cur)
			}
		}
	}
	f(0, nil)

	var r []float64
	var origIndices []int
	for _, idx := range indices {
		t2, err := t.ErrResponser().Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)

		// origIdx := 0
		// for i, id := range idx {
		// 	origIdx += id * t.Strides[i]
		// }
		// origIndices = append(origIndices, origIdx)

		combi := [][]int{}
		for i := 0; i < len(idx); i++ {
			combi = append(combi, []int{idx[i]})
		}
		if len(idx) < t.Ndim() {
			for i := len(idx); i < t.Ndim(); i++ {
				combi = append(combi, until(t.Shape[i]))
			}
		}

		rawindices := cartesians(combi)
		for _, rawidx := range rawindices {
			origIdx := t.offset
			for i := range rawidx {
				origIdx += rawidx[i] * t.Strides[i]
			}
			origIndices = append(origIndices, origIdx)
		}
	}

	newtensor, err := RespErr.NdShape(r, newshape...)
	if err != nil {
		return nil, err
	}

	return &indexResult{t: newtensor, origIndices: origIndices}, nil
}
