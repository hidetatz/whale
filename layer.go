package whale

import (
	"github.com/hidetatz/whale/tensor"
)

func Linear(x, w, b *Variable) (*Variable, error) {
	t, err := MatMul(x, w)
	if err != nil {
		return nil, err
	}
	y, err := Add(t, b)
	if err != nil {
		return nil, err
	}

	return y, nil
}

func Sigmoid(x *Variable) (*Variable, error) {
	t1, err := Neg(x)
	if err != nil {
		return nil, err
	}

	t2, err := Exp(t1)
	if err != nil {
		return nil, err
	}

	t3, err := Add(NewVar(tensor.FromScalar(1)), t2)
	if err != nil {
		return nil, err
	}

	y, err := Div(NewVar(tensor.FromScalar(1)), t3)
	if err != nil {
		return nil, err
	}

	return y, nil
}

func MeanSquaredError(x0, x1 *Variable) (*Variable, error) {
	diff, err := Sub(x0, x1)
	if err != nil {
		return nil, err
	}

	squ, err := Pow(diff, NewVar(tensor.FromScalar(2)))
	if err != nil {
		return nil, err
	}

	sm, err := Sum(squ, false)
	if err != nil {
		return nil, err
	}

	return Div(sm, NewVar(tensor.FromScalar(float64(diff.Len()))))
}
