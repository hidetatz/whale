package tensor2

import (
	"slices"
)

func (t *Tensor) advancedIndex(args ...*IndexArg) (*Tensor, error) {
	/*
	 * determine new shape.
	 */
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

	broadcastedshape, err := CanBroadcast(ts)
	if err != nil {
		return nil, err
	}

	var newshape []int

	if !slices.ContainsFunc(args, func(a *IndexArg) bool { return a.typ == _slice }) {
		// if there's no slice in args, the shape follows advanced indexing rule:
		//     In general, the shape of the resultant array will be the concatenation of the shape of
		//     the index array (or the shape that all the index arrays were broadcast to)
		//     with the shape of any unused dimensions (those not indexed) in the array being indexed.
		//     (https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
		newshape = append(broadcastedshape, t.Shape[len(args):]...)
	} else {
		// else, this is advanced and basic "mixed" indexing.
		// https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
		var separated bool
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
					separated = true
					break
				}
			}

			break
		}

		if separated {
			newshape = broadcastedshape
			for _, arg := range args {
				if arg.typ != _slice {
					continue
				}
				newshape = append(newshape, arg.s.size())
			}
			newshape = append(newshape, t.Shape[len(args):]...)
		} else {
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

			for i := tensorstart + 1; i < len(args); i++ {
				if args[i].typ == _tensor || args[i].typ == _int {
					continue
				}

				newshape = append(newshape, args[i].s.size())
			}
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

	idxargs := cartesiansIdx(indices)

	var r []float64
	for _, arg := range idxargs {
		t2, err := t.Index(arg...)
		if err != nil {
			return nil, err
		}
		r = append(r, t2.Flatten()...)
	}

	return NdShape(r, newshape...)
}
