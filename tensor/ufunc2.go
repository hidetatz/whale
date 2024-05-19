package tensor

import (
	"fmt"
	"math"
)

var (
	ADD = &universalfunc2{fn: func(f1, f2 float32) float32 { return f1 + f2 }}
	SUB = &universalfunc2{fn: func(f1, f2 float32) float32 { return f1 - f2 }}
	MUL = &universalfunc2{fn: func(f1, f2 float32) float32 { return f1 * f2 }}
	DIV = &universalfunc2{fn: func(f1, f2 float32) float32 { return f1 / f2 }}
	POW = &universalfunc2{fn: func(f1, f2 float32) float32 { return float32(math.Pow(float64(f1), float64(f2))) }}
)

type universalfunc2 struct {
	fn func(f1, f2 float32) float32
}

func (u *universalfunc2) Do(t, t2 *Tensor) (*Tensor, error) {
	nt, nt2, err := RespErr.Broadcast(t, t2)
	if err != nil {
		return nil, err
	}

	d := make([]float32, nt.Size())

	t1iter := nt.Iterator()
	t2iter := nt2.Iterator()
	for t1iter.HasNext() {
		i, v1 := t1iter.Next()
		_, v2 := t2iter.Next()

		d[i] = u.fn(v1, v2)
	}

	return RespErr.NdShape(d, copySlice(nt.Shape)...)
}

func (u *universalfunc2) At(x *Tensor, indices []*IndexArg, target *Tensor) error {
	if err := indexargcheck(x, indices); err != nil {
		return err
	}

	r, err := x.index(indices...)
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
		x.data[idx] = u.fn(x.data[idx], tg)
	}

	return nil
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Add(t2))
}

func (er *tensorErrResponser) Add(t2 *Tensor) (*Tensor, error) {
	return ADD.Do(er.t, t2)
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Sub(t2))
}

func (er *tensorErrResponser) Sub(t2 *Tensor) (*Tensor, error) {
	return SUB.Do(er.t, t2)
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Mul(t2))
}

func (er *tensorErrResponser) Mul(t2 *Tensor) (*Tensor, error) {
	return MUL.Do(er.t, t2)
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Div(t2))
}

func (er *tensorErrResponser) Div(t2 *Tensor) (*Tensor, error) {
	return DIV.Do(er.t, t2)
}

func (t *Tensor) Pow(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Pow(t2))
}

func (er *tensorErrResponser) Pow(t2 *Tensor) (*Tensor, error) {
	return POW.Do(er.t, t2)
}
