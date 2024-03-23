package tensor2

import (
	"fmt"
	"slices"
)

func (t *Tensor) IndexUpdate(fn func(float64) float64, args ...*IndexArg) error {
	if t.IsScalar() {
		return fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(args) == 0 {
		return fmt.Errorf("index accessor must not be empty")
	}

	if t.Ndim() < len(args) {
		return fmt.Errorf("too many index accessors specified: %v", args)
	}

	// if argument contains at least 1 tensor, advanced indexing will be applied.
	advanced := slices.ContainsFunc(args, func(arg *IndexArg) bool { return arg.typ == _tensor })

	if advanced {
		// return t.advancedIndex(args...)
	}

	return t.basicIndexUpdate(fn, args...)
}

func (t *Tensor) IndexSet(f float64, args ...*IndexArg) error {
	return t.IndexUpdate(func(_ float64) float64 { return f }, args...)
}

func (t *Tensor) IndexAdd(f float64, args ...*IndexArg) error {
	return t.IndexUpdate(func(f2 float64) float64 { return f2 + f }, args...)
}

func (t *Tensor) IndexSub(f float64, args ...*IndexArg) error {
	return t.IndexUpdate(func(f2 float64) float64 { return f2 - f }, args...)
}

func (t *Tensor) IndexMul(f float64, args ...*IndexArg) error {
	return t.IndexUpdate(func(f2 float64) float64 { return f2 * f }, args...)
}

func (t *Tensor) IndexDiv(f float64, args ...*IndexArg) error {
	return t.IndexUpdate(func(f2 float64) float64 { return f2 / f }, args...)
}

func (t *Tensor) basicIndexUpdate(fn func(float64) float64, args ...*IndexArg) error {
	t2, err := t.basicIndex(args...)
	if err != nil {
		return err
	}

	for _, idx := range t2.rawIndices() {
		t.data[idx] = fn(t.data[idx])
	}

	return nil
}
