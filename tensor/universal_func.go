package tensor

import (
	"math"
)

var (
	EXP  = &universalfunc1{fn: func(f float64) float64 { return math.Exp(f) }}
	NEG  = &universalfunc1{fn: func(f float64) float64 { return -f }}
	SIN  = &universalfunc1{fn: func(f float64) float64 { return math.Sin(f) }}
	COS  = &universalfunc1{fn: func(f float64) float64 { return math.Cos(f) }}
	TANH = &universalfunc1{fn: func(f float64) float64 { return math.Tanh(f) }}
	LOG  = &universalfunc1{fn: func(f float64) float64 { return math.Log(f) }}
)

func (t *Tensor) Exp() *Tensor {
	return EXP.Do(t)
}

func (t *Tensor) Neg() *Tensor {
	return NEG.Do(t)
}

func (t *Tensor) Sin() *Tensor {
	return SIN.Do(t)
}

func (t *Tensor) Cos() *Tensor {
	return COS.Do(t)
}

func (t *Tensor) Tanh() *Tensor {
	return TANH.Do(t)
}

func (t *Tensor) Log() *Tensor {
	return LOG.Do(t)
}

type universalfunc1 struct {
	fn func(f float64) float64
}

func (u *universalfunc1) Do(t *Tensor) *Tensor {
	d := make([]float64, t.Size())

	iter := t.Iterator()
	for iter.HasNext() {
		i, v := iter.Next()
		d[i] = u.fn(v)
	}

	return NdShape(d, copySlice(t.Shape)...)
}
