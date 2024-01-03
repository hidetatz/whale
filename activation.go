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

// SoftMax implements softmax for multi dimensional tensor.
type SoftMax struct {
	axis []int
}

// NewSoftMaxWithAxis initializes SoftMax activation with axis specified.
// This should be used when you want to customize the axis to apply the softmax.
func NewSoftMaxWithAxis(axis ...int) *SoftMax {
	return &SoftMax{axis: axis}
}

// NewSoftMax initializes SoftMax activation.
func NewSoftMax() *SoftMax {
	return &SoftMax{axis: []int{1}}
}

func (s *SoftMax) Activate(x *Variable) (*Variable, error) {
	y, err := Exp(x)
	if err != nil {
		return nil, err
	}

	sum, err := Sum(y, true, s.axis...)
	if err != nil {
		return nil, err
	}

	d, err := Div(y, sum)
	if err != nil {
		return nil, err
	}

	return d, nil
}
