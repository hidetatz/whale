package tensor2

import (
	"fmt"
)

// IndexSet does t[args] = target.
func (t *Tensor) IndexSet(args []*IndexArg, target *Tensor) {
	MustDo(t.ErrResponser().IndexSet(args, target))
}

func (er *tensorErrResponser) IndexSet(args []*IndexArg, target *Tensor) error {
	return er.t.indexPut(args, func(_, arg float64) float64 { return arg }, target)
}

// IndexAdd does t[args] += target.
func (t *Tensor) IndexAdd(args []*IndexArg, target *Tensor) {
	MustDo(t.ErrResponser().IndexAdd(args, target))
}

func (er *tensorErrResponser) IndexAdd(args []*IndexArg, target *Tensor) error {
	return er.t.indexPut(args, func(orig, arg float64) float64 { return orig + arg }, target)
}

// IndexSub does t[args] -= target.
func (t *Tensor) IndexSub(args []*IndexArg, target *Tensor) {
	MustDo(t.ErrResponser().IndexSub(args, target))
}

func (er *tensorErrResponser) IndexSub(args []*IndexArg, target *Tensor) error {
	return er.t.indexPut(args, func(orig, arg float64) float64 { return orig - arg }, target)
}

// IndexMul does t[args] *= target.
func (t *Tensor) IndexMul(args []*IndexArg, target *Tensor) {
	MustDo(t.ErrResponser().IndexMul(args, target))
}

func (er *tensorErrResponser) IndexMul(args []*IndexArg, target *Tensor) error {
	return er.t.indexPut(args, func(orig, arg float64) float64 { return orig * arg }, target)
}

// IndexDiv does t[args] /= target.
func (t *Tensor) IndexDiv(args []*IndexArg, target *Tensor) {
	MustDo(t.ErrResponser().IndexDiv(args, target))
}

func (er *tensorErrResponser) IndexDiv(args []*IndexArg, target *Tensor) error {
	return er.t.indexPut(args, func(orig, arg float64) float64 { return orig / arg }, target)
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

	r, err := t.indexForWrite(args...)
	if err != nil {
		return err
	}

	tgt, err := target.ErrResponser().BroadcastTo(r.t.Shape...)
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
