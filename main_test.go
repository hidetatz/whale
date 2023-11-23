package main

import (
	"math"
	"testing"
)

func TestGrad(t *testing.T) {
	x := NewVar(0.5)
	a := square(x)
	b := exp(a)
	y := square(b)

	y.Backward()

	f := func(v *Variable) *Variable {
		a2 := square(v)
		b2 := exp(a2)
		y2 := square(b2)
		return y2
	}

	x2 := NewVar(0.5)
	g2 := numDiff(f, x2, 1e-4)
	delta := math.Abs(*x.grad - g2)
	if delta > 1e-6 {
		t.Errorf("backward: %v, numerical diff: %v", *x.grad, g2)
	}
}
