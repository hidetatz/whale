package tensor2

import "fmt"

var (
	ADD *add
)

var (
	_ universalfunc = &add{}
)

type universalfunc interface {
	At(x *Tensor, indices []*IndexArg, target *Tensor) error
}

type add struct{}

func (a *add) At(x *Tensor, indices []*IndexArg, target *Tensor) error {
	return ufuncAt(x, indices, func(orig, arg float64) float64 { return orig + arg }, target)
}

func ufuncAt(x *Tensor, indices []*IndexArg, fn func(orig, arg float64) float64, target *Tensor) error {
	if x.IsScalar() {
		return fmt.Errorf("index is not defined on scalar %v", x)
	}

	if len(indices) == 0 {
		return fmt.Errorf("index accessor must not be empty")
	}

	if x.Ndim() < len(indices) {
		return fmt.Errorf("too many index accessors specified: %v", indices)
	}

	r, err := x.indexForWrite(indices...)
	if err != nil {
		return err
	}

	tgt, err := target.ErrResponser().BroadcastTo(r.t.Shape...)
	if err != nil {
		return fmt.Errorf("operands could not broadcast together with shapes %v, %v", target.Shape, r.t.Shape)
	}

	it := tgt.Iterator()
	for it.HasNext() {
		i, tg := it.Next()
		idx := r.origIndices[i]
		x.data[idx] = fn(x.data[idx], tg)
	}

	return nil
}
