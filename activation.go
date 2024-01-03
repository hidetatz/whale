package whale

import "github.com/hidetatz/whale/tensor"

type Activation interface {
	Activate(x *Variable) (*Variable, error)
}

// Sigmoid implements sigmoid function.
type Sigmoid struct{}

// NewSigmoid initializes sigmoid activation.
func NewSigmoid() *Sigmoid {
	return &Sigmoid{}
}

func (s *Sigmoid) Activate(x *Variable) (*Variable, error) {
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
