package whale

import tensor "github.com/hidetatz/whale/tensor2"

type Optimizer interface {
	Optimize(v *Variable) error
}

type SGD struct {
	learnRate *tensor.Tensor
}

func NewSGD(learnRate float64) *SGD {
	return &SGD{learnRate: tensor.Scalar(learnRate)}
}

func (s *SGD) Optimize(v *Variable) error {
	delta, err := v.GetGrad().GetData().Mul(s.learnRate)
	if err != nil {
		return err
	}

	newData, err := v.GetData().Sub(delta)
	if err != nil {
		return err
	}

	v.SetData(newData)
	return nil
}

type MomentumSGD struct {
	learnRate  *tensor.Tensor
	momentum   *tensor.Tensor
	velocities map[*Variable]*tensor.Tensor
}

func NewMomentumSGD(learnRate, momentum float64) *MomentumSGD {
	return &MomentumSGD{
		learnRate:  tensor.Scalar(learnRate),
		momentum:   tensor.Scalar(momentum),
		velocities: make(map[*Variable]*tensor.Tensor),
	}
}

func (s *MomentumSGD) Optimize(v *Variable) error {
	if _, ok := s.velocities[v]; !ok {
		s.velocities[v] = tensor.ZerosLike(v.GetData())
	}
	velocity := s.velocities[v]
	velocity, err := velocity.Mul(s.momentum)
	if err != nil {
		return err
	}

	delta, err := s.learnRate.Mul(v.GetGrad().GetData())
	if err != nil {
		return err
	}

	velocity, err = velocity.Sub(delta)
	if err != nil {
		return err
	}

	newv, err := v.GetData().Add(velocity)
	if err != nil {
		return err
	}

	v.SetData(newv)
	return nil
}
