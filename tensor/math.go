package tensor

import "math"

func (t *Tensor) apply1(fn func(f float64) float64) *Tensor {
	d := make([]float64, t.Size())

	iter := t.Iterator()
	for iter.HasNext() {
		i, v := iter.Next()
		d[i] = fn(v)
	}

	return NdShape(d, copySlice(t.Shape)...)
}

func (t *Tensor) apply2(t2 *Tensor, fn func(f1, f2 float64) float64) (*Tensor, error) {
	nt, nt2, err := RespErr.Broadcast(t, t2)
	if err != nil {
		return nil, err
	}

	d := make([]float64, nt.Size())

	t1iter := nt.Iterator()
	t2iter := nt2.Iterator()
	for t1iter.HasNext() {
		i, v1 := t1iter.Next()
		_, v2 := t2iter.Next()

		d[i] = fn(v1, v2)
	}

	return RespErr.NdShape(d, copySlice(nt.Shape)...)
}

// apply1 operations
func (t *Tensor) Exp() *Tensor {
	return t.apply1(func(f float64) float64 { return math.Exp(f) })
}

func (t *Tensor) Neg() *Tensor {
	return t.apply1(func(f float64) float64 { return -f })
}

func (t *Tensor) Sin() *Tensor {
	return t.apply1(func(f float64) float64 { return math.Sin(f) })
}

func (t *Tensor) Cos() *Tensor {
	return t.apply1(func(f float64) float64 { return math.Cos(f) })
}

func (t *Tensor) Tanh() *Tensor {
	return t.apply1(func(f float64) float64 { return math.Tanh(f) })
}

func (t *Tensor) Log() *Tensor {
	return t.apply1(func(f float64) float64 { return math.Log(f) })
}

// apply2 operations

func (t *Tensor) Pow(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Pow(t2))
}

func (er *tensorErrResponser) Pow(t2 *Tensor) (*Tensor, error) {
	return er.t.apply2(t2, func(f1, f2 float64) float64 { return math.Pow(f1, f2) })
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Add(t2))
}

func (er *tensorErrResponser) Add(t2 *Tensor) (*Tensor, error) {
	return er.t.apply2(t2, func(f1, f2 float64) float64 { return f1 + f2 })
}

func (t *Tensor) Sub(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Sub(t2))
}

func (er *tensorErrResponser) Sub(t2 *Tensor) (*Tensor, error) {
	return er.t.apply2(t2, func(f1, f2 float64) float64 { return f1 - f2 })
}

func (t *Tensor) Mul(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Mul(t2))
}

func (er *tensorErrResponser) Mul(t2 *Tensor) (*Tensor, error) {
	return er.t.apply2(t2, func(f1, f2 float64) float64 { return f1 * f2 })
}

func (t *Tensor) Div(t2 *Tensor) *Tensor {
	return MustGet(t.ErrResponser().Div(t2))
}

func (er *tensorErrResponser) Div(t2 *Tensor) (*Tensor, error) {
	return er.t.apply2(t2, func(f1, f2 float64) float64 { return f1 / f2 })
}
