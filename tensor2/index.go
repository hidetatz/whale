package tensor2

import (
	"slices"
)

func (t *Tensor) Index(args ...*IndexArg) (*Tensor, error) {
	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })

	if advanced {
		return t.advancedIndex(args...)
	}

	return t.basicIndex(args...)
}
