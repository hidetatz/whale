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

type SoftmaxCrossEntropy struct{}

func NewSoftmaxCrossEntropy() *SoftmaxCrossEntropy {
	return &SoftmaxCrossEntropy{}
}

func (s *SoftmaxCrossEntropy) Calculate(x, t *Variable) (*Variable, error) {
	n := x.data.Shape[0]
	a, err := NewSoftMax().Activate(x)
	if err != nil {
		return nil, err
	}

	p, err := Clip(a, 1e-15, 1.0)
	if err != nil {
		return nil, err
	}

	logp, err := Log(p)
	if err != nil {
		return nil, err
	}

	ar, err := tensor.Arange(0, float64(n), 1, n)
	if err != nil {
		return nil, err
	}

	tlogp, err := Index(logp, NewVar(ar), t)
	if err != nil {
		return nil, err
	}

	sum, err := Sum(tlogp, false)
	if err != nil {
		return nil, err
	}

	m, err := Mul(NewVar(tensor.Scalar(-1)), sum)
	if err != nil {
		return nil, err
	}

	d, err := Div(m, NewVar(tensor.Scalar(float64(n))))
	if err != nil {
		return nil, err
	}

	return d, nil
}
