package tensor

import (
	"math"
)

var (
	EXP  = &universalfunc1{fn: func(f float32) float32 { return float32(math.Exp(float64(f))) }}
	NEG  = &universalfunc1{fn: func(f float32) float32 { return -f }}
	SIN  = &universalfunc1{fn: func(f float32) float32 { return float32(math.Sin(float64(f))) }}
	COS  = &universalfunc1{fn: func(f float32) float32 { return float32(math.Cos(float64(f))) }}
	TANH = &universalfunc1{fn: func(f float32) float32 { return float32(math.Tanh(float64(f))) }}
	LOG  = &universalfunc1{fn: func(f float32) float32 { return float32(math.Log(float64(f))) }}
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
	fn func(f float32) float32
}

func (u *universalfunc1) Do(t *Tensor) *Tensor {
	d := make([]float32, t.Size())

	iter := t.Iterator()
	for iter.HasNext() {
		i, v := iter.Next()
		d[i] = u.fn(v)
	}

	return NdShape(d, copySlice(t.Shape)...)
}
