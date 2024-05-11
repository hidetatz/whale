package tensor2

import (
	"fmt"
)

func (t *Tensor) IndexSet(args []*IndexArg, target *Tensor) error {
	return t.indexPut(args, func(_, arg float64) float64 { return arg }, target)
}

func (t *Tensor) IndexAdd(args []*IndexArg, target *Tensor) error {
	return t.indexPut(args, func(orig, arg float64) float64 { return orig + arg }, target)
}

func (t *Tensor) IndexSub(args []*IndexArg, target *Tensor) error {
	return t.indexPut(args, func(orig, arg float64) float64 { return orig - arg }, target)
}

func (t *Tensor) IndexMul(args []*IndexArg, target *Tensor) error {
	return t.indexPut(args, func(orig, arg float64) float64 { return orig * arg }, target)
}

func (t *Tensor) IndexDiv(args []*IndexArg, target *Tensor) error {
	return t.indexPut(args, func(orig, arg float64) float64 { return orig / arg }, target)
}

func (t *Tensor) indexPut(args []*IndexArg, fn func(orig, arg float64) float64, target *Tensor) error {
	if t.IsScalar() {
		return fmt.Errorf("index is not defined on scalar %v", t)
	}

	if len(args) == 0 {
		return fmt.Errorf("index accessor must not be empty")
	}

	if t.Ndim() < len(args) {
		return fmt.Errorf("too many index accessors specified: %v", args)
	}

	r, err := t.index(args...)
	if err != nil {
		return err
	}

	tgt, err := target.BroadcastTo(r.t.Shape...)
	if err != nil {
		return fmt.Errorf("operands could not broadcast together with shapes %v, %v", target.Shape, r.t.Shape)
	}

	c := t.Copy()

	it := tgt.Iterator()
	for it.HasNext() {
		i, tg := it.Next()
		idx := r.origIndices[i]
		t.data[idx] = fn(c.data[idx], tg)
	}

	return nil
}
