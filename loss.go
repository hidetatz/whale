package whale

import "github.com/hidetatz/whale/tensor"

type LossCalculator interface {
	Calculate(pred, actual *Variable) (*Variable, error)
}

type MSE struct{}

func NewMSE() *MSE {
	return &MSE{}
}

func (m *MSE) Calculate(pred, actual *Variable) (*Variable, error) {
	diff, err := Sub(pred, actual)
	if err != nil {
		return nil, err
	}

	squ, err := Pow(diff, NewVar(tensor.Scalar(2)))
	if err != nil {
		return nil, err
	}

	sm, err := Sum(squ, false)
	if err != nil {
		return nil, err
	}

	return Div(sm, NewVar(tensor.Scalar(float64(diff.Len()))))
}
