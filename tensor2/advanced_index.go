package tensor2

import (
	"slices"
)

func (t *Tensor) advancedIndex(args ...*IndexArg) (*Tensor, error) {
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
	var r []float64
	for _, idx := range indices {
		t2, err := t.Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)
	}

	return NdShape(r, newshape...)
}

func (t *Tensor) advancedAndBasicCombinedIndex(args ...*IndexArg) (*Tensor, error) {
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

	// make sure if tensor/int argument is "separated" by slice.
	// https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
	isseparated := func(args []*IndexArg) bool {
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

	if isseparated(args) {
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
		var tensorstart int
		for i := 0; i < len(args); i++ {
			if args[i].typ == _tensor || args[i].typ == _int {
				tensorstart = i
				break
			}
			if args[i].typ == _slice {
				newshape = append([]int{args[i].s.size()}, newshape...)
			}
		}

		newshape = append(newshape, broadcastedshape...)

		for i := tensorstart + 1; i < len(args); i++ {
			if args[i].typ == _tensor || args[i].typ == _int {
				continue
			}

			newshape = append(newshape, args[i].s.size())
		}
	}

	/*
	 * pick up values.
	 */

	indices := make([][]int, len(args))
	for i, arg := range args {
		switch arg.typ {
		case _int:
			indices[i] = []int{arg.i}
		case _slice:
			if err := arg.s.tidy(t.Shape[i]); err != nil {
				return nil, err
			}
			indices[i] = arg.s.indices()
		case _tensor:
			indices[i] = toint(arg.t.Flatten())
		}
	}

	var r []float64
	for _, idx := range cartesians(indices) {
		t2, err := t.Index(intsToIndices(idx)...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)
	}

	return NdShape(r, newshape...)
}
